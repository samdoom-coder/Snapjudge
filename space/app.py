"""SnapJudge Arena (Hugging Face Space): the model plays Snake and Tic-Tac-Toe by itself, live."""
import os
import random
import sys
import time

import gradio as gr
from PIL import Image, ImageDraw

MODEL_ID = os.environ.get("MODEL_ID", "Brutalsky111/Snapjudge")

print(f"downloading {MODEL_ID}...", flush=True)
from huggingface_hub import snapshot_download
SNAP = snapshot_download(MODEL_ID)
sys.path.insert(0, SNAP)

from snapjudge.agent import GameAgent
from snapjudge.router import GameRouter
from snapjudge.data_gen import questions_for

print("loading SnapJudge...", flush=True)
DEVICE = os.environ.get("DEVICE", "cpu")  # Spaces free tier is CPU
agent = GameAgent(model_dir=SNAP, device=DEVICE)
router = GameRouter(agent=agent)
Q_TTT = questions_for("tictactoe")
Q_SNAKE = questions_for("snake")
print("model ready.", flush=True)

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

# ---------------- snake engine ----------------
def new_snake(seed=None):
    rng = random.Random(seed)
    W = H = 10
    hx, hy = rng.randint(3, 6), rng.randint(3, 6)
    body = [(hx, hy), (hx - 1, hy), (hx - 2, hy)]
    return {"W": W, "H": H, "body": body, "score": 0, "steps": 0,
            "food": free_food(set(body), W, H, rng), "rng": rng, "alive": True}

def free_food(occ, W, H, rng):
    free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in occ]
    return rng.choice(free) if free else None

def snake_step(g, d):
    dx, dy = DIRS[d]
    hx, hy = g["body"][0]
    nx, ny = hx + dx, hy + dy
    occ = set(g["body"])
    tail = g["body"][-1]
    if not (0 <= nx < g["W"] and 0 <= ny < g["H"]) or ((nx, ny) in occ and (nx, ny) != tail):
        g["alive"] = False
        return g
    g["body"] = [(nx, ny)] + g["body"]
    if (nx, ny) == g["food"]:
        g["score"] += 1
        g["food"] = free_food(set(g["body"]), g["W"], g["H"], g["rng"])
    else:
        g["body"].pop()
    g["steps"] += 1
    return g

def draw_snake(g):
    S = 40
    img = Image.new("RGB", (g["W"] * S, g["H"] * S), (17, 24, 39))
    dr = ImageDraw.Draw(img)
    if g["food"]:
        fx, fy = g["food"]
        dr.ellipse([fx * S + 8, fy * S + 8, (fx + 1) * S - 8, (fy + 1) * S - 8], fill=(239, 68, 68))
    for i, (x, y) in enumerate(g["body"]):
        c = (34, 197, 94) if i == 0 else (22, 163, 74) if i < 3 else (21, 128, 61)
        dr.rounded_rectangle([x * S + 3, y * S + 3, (x + 1) * S - 3, (y + 1) * S - 3], 8, fill=c)
    return img

def play_snake(max_steps=150, delay=0.35):
    g = new_snake()
    log = []
    yield draw_snake(g), "Game started — model is thinking…"
    while g["alive"] and g["steps"] < max_steps and g["food"]:
        st = {"game": "snake", "head": list(g["body"][0]),
              "body": [list(c) for c in g["body"]], "food": list(g["food"]), "grid": [g["W"], g["H"]]}
        r = router.predict(st, Q_SNAKE)["answers"]
        d = r["direction"]["choice"]
        conf = r["direction"]["confidence"]
        risk = r["risk"]["score"]
        log.append(f"step {g['steps']}: {d} (conf {conf:.2f}, risk {risk:.1f})")
        g = snake_step(g, d)
        status = f"{'ALIVE' if g['alive'] else 'DIED'} · step {g['steps']} · food {g['score']} · last: {d} ({conf:.2f})"
        yield draw_snake(g), status + "\n" + "\n".join(log[-8:])
        time.sleep(delay)
    yield draw_snake(g), (f"GAME OVER · survived {g['steps']} steps · ate {g['score']}  |  "
                          f"{'trapped!' if not g['alive'] else 'step limit'}\n" + "\n".join(log[-8:]))

# ---------------- tic-tac-toe engine ----------------
def ttt_winner(b):
    for a, c, d in LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a], (a, c, d)
    return None, None

def draw_ttt(b, win=None):
    S = 120
    img = Image.new("RGB", (3 * S, 3 * S), (245, 245, 244))
    dr = ImageDraw.Draw(img)
    for i in range(1, 3):
        dr.line([i * S, 0, i * S, 3 * S], fill=(120, 120, 120), width=4)
        dr.line([0, i * S, 3 * S, i * S], fill=(120, 120, 120), width=4)
    for i, v in enumerate(b):
        x, y = i % 3 * S, i // 3 * S
        if v == "X":
            dr.line([x + 25, y + 25, x + S - 25, y + S - 25], fill=(37, 99, 235), width=10)
            dr.line([x + S - 25, y + 25, x + 25, y + S - 25], fill=(37, 99, 235), width=10)
        elif v == "O":
            dr.ellipse([x + 22, y + 22, x + S - 22, y + S - 22], outline=(249, 115, 22), width=10)
    if win:
        pts = [((i % 3) * S + S // 2, (i // 3) * S + S // 2) for i in win]
        dr.line(pts, fill=(22, 163, 74), width=8)
    return img

def model_move(b, player):
    legal = [i for i, v in enumerate(b) if v == " "]
    st = {"game": "tictactoe", "board": b, "player": player,
          "board_str": "".join(c if c != " " else "." for c in b)}
    probs = router.predict(st, Q_TTT)["answers"]["next_move"]["probabilities"]
    ranked = sorted(legal, key=lambda i: -probs.get(f"cell{i}", 0))
    return ranked[0], probs

def play_ttt(vs_random=True, delay=0.8):
    b = [" "] * 9
    player = "X"
    rng = random.Random()
    log = []
    yield draw_ttt(b), "New game — X to move."
    while True:
        legal = [i for i, v in enumerate(b) if v == " "]
        if not legal:
            yield draw_ttt(b), "DRAW.\n" + "\n".join(log)
            return
        if vs_random and player == "O":
            m = rng.choice(legal)
            log.append(f"O (random): cell{m}")
        else:
            m, probs = model_move(b, player)
            log.append(f"{player} (model): cell{m} (p={probs[f'cell{m}']:.2f})")
        b[m] = player
        w, line = ttt_winner(b)
        if w:
            yield draw_ttt(b, line), f"{w} WINS!\n" + "\n".join(log)
            return
        if not any(v == " " for v in b):
            yield draw_ttt(b), "DRAW.\n" + "\n".join(log)
            return
        player = "O" if player == "X" else "X"
        yield draw_ttt(b), f"{player} to move…\n" + "\n".join(log[-10:])
        time.sleep(delay)

# ---------------- UI ----------------
with gr.Blocks(title="SnapJudge Arena — the model plays by itself") as demo:
    gr.Markdown("# SnapJudge Arena\n**The model plays by itself.** Each move is one forward pass: "
                "choice + risk + confidence, no search tree.\n\n"
                "Model: [Brutalsky111/Snapjudge](https://huggingface.co/Brutalsky111/Snapjudge)")
    with gr.Tab("Snake (self-play)"):
        with gr.Row():
            snake_img = gr.Image(label="Snake", interactive=False)
            snake_log = gr.Textbox(label="Status", lines=14)
        snake_btn = gr.Button("▶ Model plays Snake", variant="primary")
        snake_btn.click(play_snake, outputs=[snake_img, snake_log])
    with gr.Tab("Tic-Tac-Toe"):
        with gr.Row():
            ttt_img = gr.Image(label="Board", interactive=False)
            ttt_log = gr.Textbox(label="Moves", lines=14)
        mode = gr.Radio([("Model (X) vs Random (O)", True), ("Model vs Model", False)],
                        value=True, label="Mode")
        ttt_btn = gr.Button("▶ Play game", variant="primary")
        ttt_btn.click(play_ttt, inputs=mode, outputs=[ttt_img, ttt_log])

demo.launch()
