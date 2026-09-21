"""Train SnapJudge: supervised CE pretrain + RLCD proper-score fine-tune + temperature fit."""
import copy, json, math, os, random
import numpy as np
import torch
import torch.nn.functional as F

from .common import QTYPES, build_sequence, collate_items, proper_reward, temp_bucket, ece_score, confidence_from_probs
from .data_gen import questions_for, generate, split

QIDS = ["q0", "q1", "q2"]  # placeholder; real qids per game below
GAME_QIDS = {"tictactoe": ["next_move","danger","must_block"],
             "snake": ["direction","risk","will_die"],
             "templerun": ["action","urgency","game_over_soon"]}

def example_to_items(tok, ex, max_len=256, head_max_len=128):
    schema = questions_for(ex["game"])
    items=[]
    for qid in GAME_QIDS[ex["game"]]:
        qdef = schema[qid]
        q = {"t": qdef["type"], "ins": qdef["instructions"], "crit": qdef.get("criteria")}
        if q["t"]=="choice" and isinstance(q["crit"], list):
            q["crit"]={c:None for c in q["crit"]}
        seq, markers = build_sequence(tok, ex["state"], q, max_len, head_max_len)
        tgt = ex["targets"][qid]
        # truncate target if markers dropped (rare with our budgets)
        tgt = tgt[:len(markers)]
        s=sum(tgt); tgt=[t/s for t in tgt]
        items.append({"ids":seq,"markers":markers,"qtype":QTYPES[q["t"]],
                      "target":tgt,"label":int(np.argmax(tgt)),
                      "game":ex["game"],"qid":qid,"K":len(markers)})
    return items

def build_pool(tok, data, max_len=256, head_max_len=128):
    pool=[]
    for ex in data:
        pool.extend(example_to_items(tok, ex, max_len, head_max_len))
    return pool

def batchify(pool, bs, rng):
    idx=list(range(len(pool))); rng.shuffle(idx)
    for s in range(0,len(idx),bs):
        yield [pool[i] for i in idx[s:s+bs]]

def ce_loss(logits, target, mask):
    logp = torch.log_softmax(logits, -1)
    return -(target*logp*mask).sum(-1).mean()

@torch.no_grad()
def evaluate(agent, pool, bs=64):
    agent.model.eval()
    toks=agent.tok
    tot_loss=0; n=0; correct=[]; confs=[]
    by_game={}
    for s in range(0,len(pool),bs):
        b=collate_items([[it] for it in pool[s:s+bs]], toks.pad_token_id)
        logits,_=agent.model(b["input_ids"].to(agent.device),b["attention_mask"].to(agent.device),
                             b["marker_pos"].to(agent.device),b["marker_mask"].to(agent.device),
                             b["qtype"].to(agent.device))
        loss=ce_loss(logits,b["target"].to(agent.device),b["marker_mask"].to(agent.device))
        tot_loss+=loss.item()*len(pool[s:s+bs]); n+=len(pool[s:s+bs])
        pred=logits.argmax(-1).cpu().numpy(); lab=b["label"].numpy()
        prob=torch.softmax(logits,-1).cpu().numpy()
        for i,it in enumerate(pool[s:s+bs]):
            k=it["K"]; p=prob[i,:k]; p=p/p.sum()
            ok=float(pred[i]==lab[i]); cf=confidence_from_probs(p,k)
            correct.append(ok); confs.append(cf)
            g=it["game"]; by_game.setdefault(g,[]).append(ok)
    import numpy as _np
    correct=_np.array(correct); confs=_np.array(confs)
    acc=float(correct.mean()); ece=ece_score(confs,correct)
    gacc={g:float(_np.mean(v)) for g,v in by_game.items()}
    return {"loss":tot_loss/max(1,n),"acc":acc,"ece":ece,"by_game":gacc}

def fit_temperatures(agent, pool, bs=64):
    """Refit one temperature per (qtype, K-bucket) on val pool — mirrors Laya calibration."""
    from collections import defaultdict
    import torch as _t
    agent.model.eval(); toks=agent.tok
    # collect logits+targets per bucket
    buckets=defaultdict(list)
    with _t.no_grad():
        for s in range(0,len(pool),bs):
            chunk=pool[s:s+bs]
            b=collate_items([[it] for it in chunk], toks.pad_token_id)
            logits,_=agent.model(b["input_ids"].to(agent.device),b["attention_mask"].to(agent.device),
                                 b["marker_pos"].to(agent.device),b["marker_mask"].to(agent.device),
                                 b["qtype"].to(agent.device))
            L=logits.float().cpu(); T=b["target"]; M=b["marker_mask"]
            for i,it in enumerate(chunk):
                k=it["K"]; buckets[temp_bucket(int(it["qtype"] if isinstance(it["qtype"],(int,)) else b["qtype"][i].item()),k)].append(
                    (L[i,:k].numpy(), _t.tensor(it["target"][:k]).numpy()))
    temps={}
    for bk,pairs in buckets.items():
        best_t,best_nll=1.0,1e99
        for t in [0.25,0.5,0.75,1.0,1.5,2.0,3.0]:
            nlls=[]
            for z,tt in pairs:
                # variable-K safe: each item scored with its own length
                zz=z/max(t,1e-3); zz=zz-zz.max()
                e=_np_exp(zz); P=e/e.sum()
                nlls.append(float(-(tt*_np_log(P)).sum()))
            nll=float(sum(nlls)/max(1,len(nlls)))
            if nll<best_nll: best_nll,best_t=nll,t
        temps[bk]=best_t
    # map to [choice,score,noul] default + dict
    from .common import QTYPE_NAMES
    base=list(agent.cfg.get("temperature",[1.0,1.0,1.0]))
    # set base per type as mean of its buckets
    for qi,qn in QTYPE_NAMES.items():
        vals=[v for k,v in temps.items() if k.startswith(qn+":")]
        if vals: base[qi]=float(sum(vals)/len(vals))
    agent.cfg["temperature"]=base; agent.cfg["temperature_by_options"]=temps
    agent.temperature=base; agent.temperature_by_options=temps
    return temps

def _np_stack(x): import numpy as _n; return _n.stack(x)
def _np_exp(x): import numpy as _n; return _n.exp(x)
def _np_log(x): import numpy as _n; return _n.log(_n.clip(x,1e-12,1.0))

def train(agent, train_data, val_data, out_dir="/content/snapjudge_ckpt",
          epochs_pre=2, epochs_rl=1, bs_items=24, lr=2e-5, noise_std=0.3, w_proper=0.5, seed=0):
    import random as _r
    rng=_r.Random(seed)
    tok=agent.tok
    max_len=agent.cfg.get("max_len",256); head_max_len=agent.cfg.get("head_max_len",128)
    train_pool=build_pool(tok,train_data,max_len,head_max_len)
    val_pool=build_pool(tok,val_data,max_len,head_max_len)
    print(f"pool: {len(train_pool)} train items / {len(val_pool)} val items", flush=True)
    opt=_t_opt(agent)
    opt_fn=opt
    sched=None
    global_step=0
    for epoch in range(epochs_pre):
        agent.model.train()
        tot=0; nb=0
        for batch in batchify(train_pool,bs_items,rng):
            b=collate_items([[it] for it in batch], tok.pad_token_id)
            dev=agent.device
            logits,_=agent.model(b["input_ids"].to(dev),b["attention_mask"].to(dev),
                                 b["marker_pos"].to(dev),b["marker_mask"].to(dev),b["qtype"].to(dev))
            loss=ce_loss(logits,b["target"].to(dev),b["marker_mask"].to(dev))
            opt_fn.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(),1.0)
            opt_fn.step(); tot+=loss.item(); nb+=1; global_step+=1
        m=evaluate(agent,val_pool)
        print(f"[pre {epoch+1}/{epochs_pre}] ce={tot/max(1,nb):.4f} val_acc={m['acc']:.3f} ece={m['ece']:.3f} {m['by_game']}",flush=True)
    # RLCD stage: CE + maximise proper score under Gaussian logit noise (zero-mean exploration)
    for epoch in range(epochs_rl):
        agent.model.train(); tot=0; nb=0; rew=0
        for batch in batchify(train_pool,bs_items,rng):
            b=collate_items([[it] for it in batch], tok.pad_token_id)
            dev=agent.device
            logits,_=agent.model(b["input_ids"].to(dev),b["attention_mask"].to(dev),
                                 b["marker_pos"].to(dev),b["marker_mask"].to(dev),b["qtype"].to(dev))
            noisy=logits+torch.randn_like(logits)*noise_std
            q=torch.softmax(noisy,-1)
            r=proper_reward(q,b["target"].to(dev),b["qtype"].to(dev),b["marker_mask"].to(dev))
            # group-mean baseline per batch
            base=r.mean().detach()
            # pathwise maximise reward + keep CE anchor (GRPO-style baseline for logging)
            loss_pg=-(r-base).mean()*0.0 - r.mean()  # direct proper-score ascent
            loss_ce=ce_loss(logits,b["target"].to(dev),b["marker_mask"].to(dev))
            loss=loss_ce + w_proper*loss_pg
            opt_fn.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(),1.0)
            opt_fn.step(); tot+=loss.item(); rew+=r.mean().item(); nb+=1
        m=evaluate(agent,val_pool)
        print(f"[rlcd {epoch+1}/{epochs_rl}] loss={tot/max(1,nb):.4f} reward={rew/max(1,nb):.4f} val_acc={m['acc']:.3f} ece={m['ece']:.3f} {m['by_game']}",flush=True)
    # save raw weights BEFORE calibration so a calibration bug can never lose training
    agent.save(out_dir)
    try:
        temps=fit_temperatures(agent,val_pool)
        print("fitted temperatures:",temps,flush=True)
    except Exception as e:
        print("temperature fit failed (keeping T=1.0):", repr(e), flush=True)
        temps={}
    m=evaluate(agent,val_pool)
    print(f"final val_acc={m['acc']:.3f} ece={m['ece']:.3f} {m['by_game']}",flush=True)
    agent.save(out_dir)
    with open(os.path.join(out_dir,"metrics.json"),"w") as f: json.dump(m,f,indent=2)
    return m

def _t_opt(agent, lr=2e-5):
    import torch as _t
    return _t.optim.AdamW(agent.model.parameters(), lr=lr)
