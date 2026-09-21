"""Demo SnapJudge after training: tic-tac-toe, snake, temple-run via GameRouter."""
import json, time
from snapjudge.agent import load
from snapjudge.router import GameRouter
from snapjudge.data_gen import questions_for

agent = load("/content/snapjudge_ckpt")
router = GameRouter(agent=agent)

demos = [
    ({"game":"tictactoe","board":["X","X"," ","O","O"," "," "," "," "],"player":"X","board_str":"XX.OO...."},
     questions_for("tictactoe"), "tic-tac-toe: X can win now (cell2)"),
    ({"game":"snake","head":[5,5],"body":[[5,5],[5,6],[4,6]],"food":[7,5],"grid":[10,10]},
     questions_for("snake"), "snake: head (5,5), food right"),
    ({"game":"templerun","obstacle":"gap","lane":"mid","distance":"near","speed":"fast"},
     questions_for("templerun"), "temple-run: gap near, fast"),
]
for state, qs, title in demos:
    t0=time.time()
    r=router.predict(state, qs)
    dt=(time.time()-t0)*1000
    print(f"\n=== {title} ({dt:.1f} ms, route={r['routing']['model']}: {r['routing']['reason']}) ===")
    print(json.dumps(r["answers"], indent=1))
