"""SnapJudge Agent: single-forward-pass typed decisions over game states."""
import json
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .common import QTYPES, build_model, build_sequence, collate_items, confidence_from_probs, render_options, temp_bucket


class GameAgent:
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None,
                 encoder: str = "answerdotai/ModernBERT-base"):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoTokenizer
        # new name first, old gamelaya name as fallback (backward compatible)
        cfg_path = None
        if model_dir:
            for _name in ("snapjudge_config.json", "gamelaya_config.json"):
                _p = os.path.join(model_dir, _name)
                if os.path.exists(_p):
                    cfg_path = _p
                    break
        if cfg_path:
            with open(cfg_path) as f:
                self.cfg = json.load(f)
            self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer") if os.path.exists(os.path.join(model_dir, "tokenizer")) else self.cfg.get("encoder", encoder))
            from safetensors.torch import load_file
            self.model = build_model(self.cfg)
            w = load_file(os.path.join(model_dir, "model.safetensors"))
            self.model.load_state_dict(w, strict=True)
        else:
            self.cfg = {"encoder": encoder, "head_layers": 2, "act_costs": {"escalate": 1},
                        "max_len": 256, "head_max_len": 128,
                        "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
            self.tok = AutoTokenizer.from_pretrained(encoder)
            self.model = build_model(self.cfg)
        try:
            self.model.encoder.config.reference_compile = False
        except Exception:
            pass
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.model.to(self.device).eval()

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    @torch.no_grad()
    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        ids = list(questions.keys())
        items = []
        max_len = self.cfg.get("max_len", 256)
        head_max_len = self.cfg.get("head_max_len", 128)
        for qid in ids:
            q = self._to_internal(questions[qid])
            seq, markers = build_sequence(self.tok, state, q, max_len, head_max_len)
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], self.tok.pad_token_id)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            logits, act = self.model(b["input_ids"].to(self.device), b["attention_mask"].to(self.device),
                                     b["marker_pos"].to(self.device), b["marker_mask"].to(self.device),
                                     b["qtype"].to(self.device))
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        answers = {}
        n_tokens = int(b["attention_mask"].sum())
        for r, qid in enumerate(ids):
            q = self._to_internal(questions[qid])
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / max(1e-3, float(t_scale))
            p = np.exp(z - z.max()); p = p / p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                                "confidence": conf, "action": ext}
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {"type": "score", "score": round(exp_score, 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                                "confidence": conf, "action": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
        return {"model": "snapjudge", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}}

    predict = system_one

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "snapjudge_config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        self.tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
        from safetensors.torch import save_file
        save_file(self.model.state_dict(), os.path.join(out_dir, "model.safetensors"))


def load(model_dir: str, device: Optional[str] = None) -> GameAgent:
    return GameAgent(model_dir=model_dir, device=device)
