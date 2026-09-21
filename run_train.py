from snapjudge.agent import GameAgent
from snapjudge.data_gen import generate, split
from snapjudge.train import train

print("generating data...", flush=True)
data = generate(n_per_game=2000, seed=0)
tr, va = split(data, val_frac=0.1, seed=1)
print(f"train ex={len(tr)} val ex={len(va)}", flush=True)

print("building agent...", flush=True)
agent = GameAgent(device="cuda")
print("training...", flush=True)
m = train(agent, tr, va, out_dir="/content/snapjudge_ckpt", epochs_pre=3, epochs_rl=2, bs_items=24, lr=2e-5)
print("DONE", m, flush=True)
