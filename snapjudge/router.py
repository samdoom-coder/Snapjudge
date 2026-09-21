"""GameRouter: route by game type before the forward pass (like Laya's Router routes by language)."""
from typing import Any, Dict, List, Optional, Union


class RouteDecision(dict):
    @property
    def model(self): return self["model"]
    @property
    def reason(self): return self["reason"]


KNOWN_GAMES = ("tictactoe", "snake", "templerun")


def detect_game(state) -> Dict[str, Any]:
    if isinstance(state, dict):
        g = str(state.get("game", "")).lower()
        if g in ("tic-tac-toe", "tictactoe", "ttt"):
            return {"game": "tictactoe", "confidence": 1.0}
        if g == "snake":
            return {"game": "snake", "confidence": 1.0}
        if g in ("temple_run", "templerun", "temple-run"):
            return {"game": "templerun", "confidence": 1.0}
        # heuristics
        if "board" in state and "player" in state:
            return {"game": "tictactoe", "confidence": 0.8}
        if "head" in state and "food" in state:
            return {"game": "snake", "confidence": 0.8}
        if "obstacle" in state and "distance" in state:
            return {"game": "templerun", "confidence": 0.8}
    if isinstance(state, str):
        s = state.lower()
        if "board" in s or "tictactoe" in s or "tic-tac" in s:
            return {"game": "tictactoe", "confidence": 0.6}
        if "snake" in s or "food" in s:
            return {"game": "snake", "confidence": 0.6}
        if "temple" in s or "obstacle" in s or "jump" in s:
            return {"game": "templerun", "confidence": 0.6}
    return {"game": "unknown", "confidence": 0.0}


class GameRouter:
    """Single joint checkpoint for v1 (all games), with routing metadata + per-game override support."""

    def __init__(self, agent=None, agents: Optional[Dict[str, Any]] = None, device: Optional[str] = None, default: str = "joint"):
        self.device = device
        self.default = default
        self._agents: Dict[str, Any] = {}
        if agent is not None:
            self._agents["joint"] = agent
        if agents:
            self._agents.update(agents)

    def attach(self, name: str, agent):
        self._agents[name] = agent
        return agent

    def route(self, state, questions=None, model: Optional[str] = None) -> RouteDecision:
        if model is not None:
            return RouteDecision(model=model, reason="explicit model=%r" % model, detection=None)
        det = detect_game(state)
        if det["game"] in KNOWN_GAMES:
            # v1: one joint model serves all games; reason records why
            key = "joint" if "joint" in self._agents or len(self._agents) <= 1 else (det["game"] if det["game"] in self._agents else "joint")
            return RouteDecision(model=key, reason="game=%s (conf %.1f)" % (det["game"], det["confidence"]), detection=det)
        return RouteDecision(model=self.default if self.default in self._agents else next(iter(self._agents), "joint"),
                             reason="unknown game; using default", detection=det)

    def predict(self, state, questions, model: Optional[str] = None):
        d = self.route(state, questions, model=model)
        agent = self._agents.get(d["model"])
        if agent is None:
            if not self._agents:
                raise RuntimeError("GameRouter has no agent attached. Attach a GameAgent first.")
            agent = next(iter(self._agents.values()))
            d["model"] = "fallback"
        res = agent.system_one(state, questions)
        res["routing"] = dict(d)
        return res

    system_one = predict
