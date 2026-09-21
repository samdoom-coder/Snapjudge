"""Synthetic game states + optimal targets for SnapJudge.
Three games, typed questions each (choice/score/noul) — mirrors Laya's primitives.
All labels are computed by exact rules (minimax / collision / obstacle table), so no LLM teacher needed.
"""
import random
from typing import Dict, List, Tuple

# ---------------- shared question schemas ----------------

def tictactoe_questions():
    cells = {}
    for i in range(9):
        r, c = divmod(i, 3)
        cells[f"cell{i}"] = f"row {r} col {c}"
    return {
        "next_move": {"type": "choice", "instructions": "Play the optimal tic-tac-toe move for the current player.",
                      "criteria": cells},
        "danger": {"type": "score", "instructions": "How decisive is this board position?",
                   "criteria": ["safe position", "threat building", "immediate win or must-block"]},
        "must_block": {"type": "noul", "instructions": "Must the player block an immediate opponent win?"},
    }

def snake_questions():
    return {
        "direction": {"type": "choice", "instructions": "Choose the safest next direction that approaches food.",
                      "criteria": {"up": "move up (y-1)", "down": "move down (y+1)",
                                   "left": "move left (x-1)", "right": "move right (x+1)"}},
        "risk": {"type": "score", "instructions": "How risky is this snake position?",
                 "criteria": ["safe, 2+ escape moves", "risky, only 1 safe move", "deadly, no safe moves"]},
        "will_die": {"type": "noul", "instructions": "Is the snake trapped with no safe moves?"},
    }

def templerun_questions():
    return {
        "action": {"type": "choice", "instructions": "React to the obstacle ahead in temple run.",
                   "criteria": {"jump": "jump over gap", "slide": "slide under fire",
                                "left": "dodge left", "right": "dodge right", "run": "keep running"}},
        "urgency": {"type": "score", "instructions": "How urgent is the reaction?",
                    "criteria": ["far away", "approaching", "immediate action required"]},
        "game_over_soon": {"type": "noul", "instructions": "Will the runner crash without immediate correct action?"},
    }

def questions_for(game: str):
    return {"tictactoe": tictactoe_questions, "snake": snake_questions, "templerun": templerun_questions}[game]()

# ---------------- tic-tac-toe exact logic ----------------
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def ttt_winner(b):
    for a,c,d in LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None

def ttt_legal(b): return [i for i,v in enumerate(b) if v == " "]

def ttt_immediate_wins(b, p):
    return [i for i in ttt_legal(b) if ttt_winner([p if j==i else v for j,v in enumerate(b)]) == p]

def ttt_minimax(b, player, me, memo=None):
    memo = memo if memo is not None else {}
    key = ("".join(b), player)
    if key in memo: return memo[key]
    w = ttt_winner(b)
    if w == me: return (1, None)
    if w is not None: return (-1, None)
    if not ttt_legal(b): return (0, None)
    opp = "O" if player == "X" else "X"
    best, bestm = (-2 if player==me else 2), None
    for m in ttt_legal(b):
        b2 = list(b); b2[m]=player
        s,_ = ttt_minimax(b2, opp, me, memo)
        if player==me and s>best: best,bestm=s,m
        if player!=me and s<best: best,bestm=s,m
    memo[key]=(best,bestm); return best,bestm

def random_ttt_board(rng):
    while True:
        # X starts: nx == no (O to move... actually X to move when equal) or nx == no+1 (O to move)
        no = rng.randint(0, 4)
        nx = no + rng.choice([0, 1])
        if nx < 1 or nx + no > 7:
            continue
        b = [" "] * 9
        cells = rng.sample(range(9), nx + no)
        for i in cells[:nx]: b[i]="X"
        for i in cells[nx:]: b[i]="O"
        if ttt_winner(b): continue
        player = "X" if nx==no else "O"
        if not ttt_legal(b): continue
        return b, player

def ttt_example(rng):
    b, player = random_ttt_board(rng)
    opp = "O" if player=="X" else "X"
    legal = ttt_legal(b)
    # optimal moves: all moves achieving minimax value
    vals = {}
    for m in legal:
        b2=list(b); b2[m]=player
        v,_ = ttt_minimax(b2, opp, player)
        vals[m]=v
    best_v = max(vals.values())
    best = [m for m,v in vals.items() if v==best_v]
    target = [0.0]*9
    for m in best: target[m]=1.0/len(best)
    my_win = ttt_immediate_wins(b, player)
    op_win = ttt_immediate_wins(b, opp)
    if my_win or op_win: danger=2
    elif best_v==1 or best_v==-1: danger=1
    else:
        # threat heuristic: any 2-in-row with empty third?
        danger=0
        for a,c,d in LINES:
            line=[b[a],b[c],b[d]]
            if line.count(player)==2 and line.count(" ")==1: danger=max(danger,1)
            if line.count(opp)==2 and line.count(" ")==1: danger=max(danger,1)
    must_block = bool(op_win) and not bool(my_win)
    state = {"game":"tictactoe","board":b,"player":player,
             "board_str":"".join(c if c!=" " else "." for c in b)}
    targets = {"next_move": target, "danger": [1.0 if i==danger else 0.0 for i in range(3)],
               "must_block": [0.0,1.0] if must_block else [1.0,0.0]}
    labels = {"next_move": best[0], "danger": danger, "must_block": 1 if must_block else 0}
    return state, targets, labels

# ---------------- snake ----------------
DIRS = {"up":(0,-1),"down":(0,1),"left":(-1,0),"right":(1,0)}
DIR_ORDER = ["up","down","left","right"]

def snake_example(rng, W=10, H=10):
    L = rng.randint(2,5)
    hx, hy = rng.randint(1,W-2), rng.randint(1,H-2)
    body=[(hx,hy)]
    for _ in range(L-1):
        x,y=body[-1]
        cands=[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        cands=[c for c in cands if 0<=c[0]<W and 0<=c[1]<H and c not in body]
        if not cands: break
        body.append(rng.choice(cands))
    while True:
        fx,fy=rng.randint(0,W-1),rng.randint(0,H-1)
        if (fx,fy) not in body: break
    occupied=set(body)
    safe={}
    for d in DIR_ORDER:
        dx,dy=DIRS[d]; nx,ny=hx+dx,hy+dy
        tail=body[-1]
        # moving into tail is ok (tail vacates) unless growing — ignore growth for simplicity
        hit_wall=not(0<=nx<W and 0<=ny<H)
        hit_self=(nx,ny) in occupied and (nx,ny)!=tail
        safe[d]=not(hit_wall or hit_self)
    n_safe=sum(safe.values())
    # pick safe move minimising manhattan to food
    best=None;bestd=1e9
    for d in DIR_ORDER:
        if not safe[d]: continue
        dx,dy=DIRS[d]
        dist=abs(hx+dx-fx)+abs(hy+dy-fy)
        if dist<bestd: bestd=dist;best=d
    if best is None:
        best=rng.choice(DIR_ORDER)  # trapped, arbitrary
    order_idx={d:i for i,d in enumerate(DIR_ORDER)}
    target=[0.0]*4
    # if several safe moves tie on distance, share mass
    tied=[d for d in DIR_ORDER if safe[d] and abs(hx+DIRS[d][0]-fx)+abs(hy+DIRS[d][1]-fy)==bestd]
    if tied:
        for d in tied: target[order_idx[d]]=1.0/len(tied)
    else:
        target[order_idx[best]]=1.0
    risk = 0 if n_safe>=2 else (1 if n_safe==1 else 2)
    trapped = (n_safe==0)
    state={"game":"snake","head":[hx,hy],"body":body,"food":[fx,fy],"grid":[W,H]}
    targets={"direction":target,"risk":[1.0 if i==risk else 0.0 for i in range(3)],
             "will_die":[0.0,1.0] if trapped else [1.0,0.0]}
    labels={"direction":order_idx[best],"risk":risk,"will_die":1 if trapped else 0}
    return state,targets,labels

# ---------------- temple run ----------------
OBSTACLES=["gap","fire","wall","none"]
def templerun_example(rng):
    ob=rng.choice(OBSTACLES); lane=rng.choice(["left","mid","right"])
    dist=rng.choice(["far","mid","near"]); speed=rng.choice(["slow","fast"])
    if ob=="gap": act="jump"
    elif ob=="fire": act="slide"
    elif ob=="wall": act="right" if lane=="left" else "left"
    else: act="run"
    acts=["jump","slide","left","right","run"]
    target=[1.0 if a==act else 0.0 for a in acts]
    urg={"far":0,"mid":1,"near":2}[dist]
    over = (ob!="none" and dist=="near" and speed=="fast")
    state={"game":"templerun","obstacle":ob,"lane":lane,"distance":dist,"speed":speed}
    targets={"action":target,"urgency":[1.0 if i==urg else 0.0 for i in range(3)],
             "game_over_soon":[0.0,1.0] if over else [1.0,0.0]}
    labels={"action":acts.index(act),"urgency":urg,"game_over_soon":1 if over else 0}
    return state,targets,labels

GEN = {"tictactoe": ttt_example, "snake": snake_example, "templerun": templerun_example}

def generate(n_per_game: int = 2000, seed: int = 0):
    rng=random.Random(seed)
    out=[]
    for g,fn in GEN.items():
        for _ in range(n_per_game):
            s,t,l=fn(rng)
            out.append({"game":g,"state":s,"targets":t,"labels":l})
    rng.shuffle(out)
    return out

def split(data, val_frac=0.1, seed=0):
    rng=random.Random(seed); d=list(data); rng.shuffle(d)
    n=int(len(d)*val_frac)
    return d[n:], d[:n]
