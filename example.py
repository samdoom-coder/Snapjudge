"""Runnable quickstart for the SnapJudge HF repo."""
from pathlib import Path
from snapjudge.agent import load
from snapjudge.router import GameRouter
from snapjudge.data_gen import questions_for

_HERE = Path(__file__).resolve().parent
router = GameRouter(agent=load(str(_HERE)))  # robust to cwd

demos = [
    ({"game": "tictactoe", "board": ["X", "X", " ", "O", "O", " ", " ", " ", " "],
      "player": "X", "board_str": "XX.OO...."},
     questions_for("tictactoe"), "tic-tac-toe"),
    ({"game": "snake", "head": [5, 5], "body": [[5, 5], [5, 6]], "food": [7, 5], "grid": [10, 10]},
     questions_for("snake"), "snake"),
    ({"game": "templerun", "obstacle": "gap", "lane": "mid", "distance": "near", "speed": "fast"},
     questions_for("templerun"), "temple-run"),
]
for state, qs, title in demos:
    r = router.predict(state, qs)
    print(f"=== {title} (route={r['routing']['model']}: {r['routing']['reason']}) ===")
    for qid, a in r["answers"].items():
        print(f"  {qid}: {a}")
