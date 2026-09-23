"""Professional SnapJudge performance chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

games = ["temple-run", "snake", "tic-tac-toe"]
card = [1.0, 0.9386, 0.7721]
seed999 = [1.0, 0.9983, 0.8483]
seed123 = [1.0, 0.9950, 0.8567]

qs = ["action", "urgency", "game_over", "direction", "risk", "will_die", "next_move*", "danger", "block"]
tie999 = [1.0, 1.0, 1.0, 0.995, 1.0, 1.0, 0.865, 0.885, 0.795]
conf999 = [0.996, 0.999, 0.999, 0.666, 1.0, 1.0, 0.438, 0.781, 0.814]
colors = ["#2a9d8f"]*3 + ["#218380"]*3 + ["#e76f51"]*3

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("SnapJudge — System-1 Decision Model Performance (165M, ModernBERT-base)", fontsize=14, fontweight="bold")
fig.text(0.5, 0.93, "Fresh synthetic held-out: 600 games / 1,800 decisions per seed  •  Tesla T4, fp16  •  *tie-aware (any optimal move counts)",
         ha="center", fontsize=9, color="#555")

a = ax[0, 0]
x = np.arange(len(games)); w = 0.24
a.bar(x-w, card, w, label="Model card", color="#adb5bd")
a.bar(x, seed999, w, label="Fresh seed 999", color="#264653")
a.bar(x+w, seed123, w, label="Fresh seed 123", color="#2a9d8f")
a.set_xticks(x); a.set_xticklabels(games); a.set_ylim(0.6, 1.02); a.set_ylabel("Accuracy")
a.set_title("Accuracy by game (reproduces card, slightly better)")
a.legend(frameon=False, fontsize=8)
for i in range(3):
    a.text(i-w, card[i]+0.008, f"{card[i]:.3f}", ha="center", fontsize=7, color="#666")
    a.text(i, seed999[i]+0.008, f"{seed999[i]:.3f}", ha="center", fontsize=7, fontweight="bold")
a.grid(axis="y", alpha=0.3)

b = ax[0, 1]
y = np.arange(len(qs))
b.barh(y, tie999, color=colors, alpha=0.9)
b.set_yticks(y); b.set_yticklabels(qs); b.set_xlim(0.4, 1.02); b.set_xlabel("Tie-aware accuracy (seed 999)")
b.set_title("Per-question accuracy — temple/snake perfect, ttt weakest")
for i, v in enumerate(tie999):
    b.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=8)
b.grid(axis="x", alpha=0.3)
b.invert_yaxis()

c = ax[1, 0]
c.scatter(conf999, tie999, s=90, c=colors, edgecolors="white", zorder=3)
# manual offsets to unstack the perfect cluster at (1.0, 1.0)
offsets = {
    "action": (8, 8), "urgency": (8, -12), "game_over": (-42, 8),
    "direction": (8, 5), "risk": (8, -3), "will_die": (-48, -12),
    "next_move*": (8, 3), "danger": (8, 5), "block": (8, 3),
}
for i, q in enumerate(qs):
    dx, dy = offsets.get(q, (5, 4))
    c.annotate(q, (conf999[i], tie999[i]), xytext=(dx, dy), textcoords="offset points",
               fontsize=7, ha="left" if dx >= 0 else "right",
               arrowprops=dict(arrowstyle="-", color="#888", lw=0.6, shrinkB=4))
c.plot([0.4, 1.0], [0.4, 1.0], "--", color="#999", lw=1, label="perfect calibration")
c.set_xlim(0.4, 1.02); c.set_ylim(0.75, 1.02)
c.set_xlabel("Mean confidence"); c.set_ylabel("Accuracy")
c.set_title("Calibration — confidence vs accuracy (ECE 0.096)")
c.legend(frameon=False, fontsize=8); c.grid(alpha=0.3)

d = ax[1, 1]
d.axis("off")
rows = [
    ["Overall (tie-aware)", "0.949 → 0.951", "card 0.905 (strict)"],
    ["ECE", "0.096 – 0.098", "card 0.049 (orig split)"],
    ["Latency 3-Q call", "37.7 ms p50", "card ~36 ms ✓"],
    ["Batched 10 states", "14.1 ms / state", "2.6× throughput"],
    ["Params / ctx", "165M / 256", "Laya 421M / 512"],
]
tbl = d.table(cellText=rows, colLabels=["Metric", "Measured", "Reference"],
              loc="center", colWidths=[0.38, 0.3, 0.32])
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
for (r, col), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#264653"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f1faee")
d.set_title("Summary — single forward pass, nothing to parse", pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.9])
out = "snapjudge_perf.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
