"""Bench SnapJudge like Laya BENCHMARKS.md: fresh synthetic held-out, accuracy + ECE + latency.
Run: python3 bench.py --model snapjudge_repo --n 200 --seed 999
"""
import argparse, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from snapjudge.agent import load
from snapjudge.router import GameRouter
from snapjudge.data_gen import generate, questions_for, GEN
from snapjudge.common import ece_score

GAME_QIDS = {"tictactoe": ["next_move","danger","must_block"],
             "snake": ["direction","risk","will_die"],
             "templerun": ["action","urgency","game_over_soon"]}

def run(args):
    router = GameRouter(agent=load(args.model, device=args.device))
    data = generate(n_per_game=args.n, seed=args.seed)
    print(f"generated {len(data)} games ({len(data)*3} decisions), seed={args.seed}", flush=True)
    correct, confs, by_game = [], [], {}
    strict = []
    # warmup
    s0 = data[0]["state"]; q0 = questions_for(data[0]["game"])
    for _ in range(3):
        router.predict(s0, q0)
    t0 = time.perf_counter()
    lat = []
    for ex in data:
        g = ex["game"]
        qs = questions_for(g)
        t1 = time.perf_counter()
        res = router.predict(ex["state"], qs)
        lat.append((time.perf_counter()-t1)*1000)
        for qid in GAME_QIDS[g]:
            true = ex["labels"][qid]
            tgt = np.array(ex["targets"][qid])
            ans = res["answers"][qid]
            if ans["type"] == "choice":
                keys = list(qs[qid]["criteria"].keys())
                pred = keys.index(ans["choice"])
                # multiple optimal moves share mass: any optimal move counts
                ok = float(tgt[pred] > 0)
                strict.append(float(pred == true))
                conf = ans["confidence"]
            elif ans["type"] == "score":
                probs = [ans["probabilities"][str(i)] for i in range(len(qs[qid]["criteria"]))]
                pred = int(np.argmax(probs))
                ok = float(pred == true)
                conf = ans["confidence"]
            else:
                prob_true = ans["noul"]
                pred = 1 if prob_true >= 0.5 else 0
                ok = float(pred == true)
                conf = ans["confidence"]
            correct.append(ok); confs.append(conf)
            by_game.setdefault(g, []).append(ok)
    correct = np.array(correct); confs = np.array(confs)
    acc = float(correct.mean()); ece = ece_score(confs, correct)
    print(f"overall tie-aware acc={acc:.3f} strict-choice acc={np.mean(strict):.3f} ece={ece:.3f}")
    for g, v in by_game.items():
        print(f"  {g}: {np.mean(v):.3f} ({len(v)} decisions)")
    lat = np.array(lat)
    print(f"latency per 3-question call (ms): p50={np.median(lat):.1f} mean={lat.mean():.1f} p90={np.percentile(lat,90):.1f}")
    # batched throughput: 10 questions = 3 games + extras
    qs_big = dict(list(questions_for('snake').items())*3 + [('extra', {'type':'noul','instructions':'Is this safe?'})])
    states = [data[i]['state'] for i in range(min(10, len(data)))]
    t1 = time.perf_counter()
    router._agents['joint'].predict_batch(states, [questions_for(data[i]['game']) for i in range(len(states))])
    dt = (time.perf_counter()-t1)*1000
    print(f"batched {len(states)} states in {dt:.1f} ms ({dt/max(1,len(states)):.1f} ms/state)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", default=None)
    run(ap.parse_args())
