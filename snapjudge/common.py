"""SnapJudge core: encoder + decision head, sequence building, RLCD rewards.
Full-replica of Laya idea (convaiinnovations/laya) adapted for game states.
Backbone default: answerdotai/ModernBERT-base (149M) for T4-friendly training.
"""
import json
import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

DEFAULT_ENCODER = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LEN = 256
DEFAULT_HEAD_MAX_LEN = 128


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(tok, state, q: Dict, max_len: int = DEFAULT_MAX_LEN,
                   head_max_len: int = DEFAULT_HEAD_MAX_LEN,
                   option_order: Optional[List[int]] = None):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]."""
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:32]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
    head_ids = head_ids[: max(8, head_max_len - sum(len(o) for o in opt_ids))]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


class DecisionModel(nn.Module):
    """ModernBERT encoder + 2-layer transformer decision head + option scorer + act head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_model(cfg: Dict, encoder_dir: Optional[str] = None) -> DecisionModel:
    from transformers import AutoConfig, AutoModel
    enc_name = cfg.get("encoder", DEFAULT_ENCODER)
    if encoder_dir:
        import os
        if os.path.exists(encoder_dir):
            ecfg = AutoConfig.from_pretrained(encoder_dir)
            enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
            return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    enc = AutoModel.from_pretrained(enc_name, attn_implementation="sdpa")
    try:
        if getattr(enc.config, "reference_compile", None):
            enc.config.reference_compile = False
    except Exception:
        pass
    return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)


def proper_reward(q, target, qtype, mask, w_sph: float = 0.5, w_rps: float = 1.0, log_floor: float = -9.21):
    """Strictly proper scoring rule: log + spherical + RPS for ordinal. Maximised only by honest probs."""
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    import torch as _t
    ids = _t.full((n, L), pad_id, dtype=_t.long)
    att = _t.zeros((n, L), dtype=_t.long)
    mpos = _t.zeros((n, kmax), dtype=_t.long)
    mmask = _t.zeros((n, kmax), dtype=_t.bool)
    has_target = any("target" in it for it in items)
    target = _t.zeros((n, kmax), dtype=_t.float32) if has_target else None
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = _t.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = _t.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            target[i, : len(it["target"])] = _t.tensor(it["target"], dtype=_t.float32)
    res = {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "qtype": _t.tensor([it["qtype"] for it in items]),
        "label": _t.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target")} for it in items],
    }
    if target is not None:
        res["target"] = target
    return res
