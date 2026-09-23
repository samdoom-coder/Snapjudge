"""Play tic-tac-toe vs SnapJudge. Tests next_move + danger + must_block live.
Run: python3 play_ttt.py [--human X|O] [--model .]
Non-interactive demo: python3 play_ttt.py --demo (AI vs AI)
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snapjudge.agent import load
from snapjudge.router import GameRouter
from snapjudge.data_gen import questions_for, ttt_winner, ttt_legal

def show(b):
    c = [x if x != " " else "." for x in b]
    print(f" {c[0]} | {c[1]} | {c[2]}\n---+---+---\n {c[3]} | {c[4]} | {c[5]}\n---+---+---\n {c[6]} | {c[7]} | {c[8]}")

def ai_move(router, board, player):
    state = {"game": "tictactoe", "board": board, "player": player,
             "board_str": "".join(x if x != " " else "." for x in board)}
    res = router.predict(state, questions_for("tictactoe"))
    a = res["answers"]
    # pick best LEGAL move (model may rank illegal cells)
    probs = a["next_move"]["probabilities"]
    legal = ttt_legal(board)
    mv = max(legal, key=lambda i: probs[f"cell{i}"])
    return mv, a

def play(human="X", model=".", device=None, demo=False):
    router = GameRouter(agent=load(model, device=device))
    board = [" "] * 9
    turn = "X"
    show(board)
    while True:
        w = ttt_winner(board)
        if w:
            print(f"Winner: {w}"); return w
        if not ttt_legal(board):
            print("Draw"); return "draw"
        if demo or turn != human:
            mv, a = ai_move(router, board, turn)
            print(f"AI({turn}) -> cell{mv} conf={a['next_move']['confidence']} "
                  f"danger={a['danger']['probabilities']} block_p={a['must_block']['noul']:.2f}")
            board[mv] = turn
        else:
            legal = ttt_legal(board)
            while True:
                try:
                    mv = int(input(f"You({human}) legal {legal}: "))
                    if mv in legal: break
                except Exception:
                    pass
                print("invalid, try again")
            board[mv] = turn
        show(board)
        turn = "O" if turn == "X" else "X"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="O", choices=["X", "O"])
    ap.add_argument("--model", default=".")
    ap.add_argument("--device", default=None)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    play(a.human, a.model, a.device, a.demo)
