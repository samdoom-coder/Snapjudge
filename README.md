# SnapJudge

Non-autoregressive System-1 decision model for games — built with the same
ideas as [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
(option-marker scoring, RLCD proper-score training, calibration, routing),
retrained from scratch for game states on a ModernBERT-base backbone.

Give it a game state (tic-tac-toe, snake, temple-run) and typed questions —
it answers everything in **one forward pass** (~36 ms warm on T4) with
calibrated probabilities. No text generation: nothing to parse, nothing to hallucinate.

**Weights & model card:** https://huggingface.co/Brutalsky111/Snapjudge

## Repo layout

```
snapjudge/        # runtime: common.py (model+head+RLCD), agent.py, router.py, data_gen.py, train.py
run_train.py     # training entrypoint (CE pretrain + RLCD + temperature fit)
demo_snapjudge.py # 3-game demo via GameRouter
space/           # Gradio arena app (app.py + requirements.txt) for HF Spaces
```

## Quickstart (weights from Hub)

```bash
pip install -r requirements.txt
git clone https://huggingface.co/Brutalsky111/Snapjudge snapjudge-weights
```

```python
from snapjudge.agent import load
from snapjudge.router import GameRouter
from snapjudge.data_gen import questions_for

router = GameRouter(agent=load("snapjudge-weights"))
res = router.predict(
    {"game": "snake", "head": [5, 5], "body": [[5, 5], [5, 6]], "food": [7, 5], "grid": [10, 10]},
    questions_for("snake"))
print(res["answers"]["direction"]["choice"])  # -> right
```

## Retrain

```bash
python3 run_train.py       # needs a T4+ GPU; ~1h for 5,400 games
python3 demo_snapjudge.py
```

Held-out: **acc 0.905, ECE 0.049** (temple-run 1.00, snake 0.939, tic-tac-toe 0.772).
See the [model card](https://huggingface.co/Brutalsky111/Snapjudge) for architecture,
training curve, speed table and honest limits.

## Try it live (Gradio arena)

Watch the model play Snake and Tic-Tac-Toe by itself — no install needed:

- **Live demo:** https://huggingface.co/spaces/Brutalsky111/Snapjudge-arena

Deploy it yourself with the 3 files in `space/` (`app.py`, `requirements.txt`, `README.md`):
1. Create a Space at https://huggingface.co/new-space — Gradio SDK, CPU, Public.
2. Upload the 3 files via the Space's Files page.
3. Wait for the build — the app downloads the weights from the model repo automatically.

Or run the arena locally:

```bash
pip install -r space/requirements.txt
python3 space/app.py
```

## License

Apache 2.0. Methodology inspired by Laya (Convai Innovations, Apache 2.0).
