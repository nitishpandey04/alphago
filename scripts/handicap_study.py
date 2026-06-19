"""Study how a per-side MCTS-simulation handicap shifts the game's outcome.

White's simulation budget is fixed; Black's is swept over a fine grid. For each
Black budget we play many games with randomized openings (temperature on the
first ``temp_moves`` moves, then greedy) and measure Black's win-rate and mean
score margin. This locates the crossover where deeper White search overcomes
Black's first-move advantage -- all from a *single* checkpoint.

Run with:
  uv run python -m scripts.handicap_study --white-sims 64 --games 16
Outputs a CSV and a PNG under --out (default: study/).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from go_agent.config import load_config
from go_agent.game import Board, BLACK
from go_agent.model import PolicyValueNet
from go_agent.mcts import search


def load_net(path, cfg, device):
    net = PolicyValueNet(
        board_size=cfg["game"]["board_size"],
        in_channels=cfg["model"]["in_channels"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        channels=cfg["model"]["channels"],
        value_channels=cfg["model"]["value_channels"],
        hidden_size=cfg["model"]["hidden_size"],
    ).to(device)
    if path and os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def play_game(net, device, cfg, sims_b, sims_w, temp_moves, seed):
    """One game; Black uses sims_b, White uses sims_w. Returns Black's score."""
    m = cfg["mcts"]
    komi = cfg["game"]["komi"]
    rng = np.random.default_rng(seed)
    b = Board(cfg["game"]["board_size"])
    mv = 0
    while not b.is_terminal() and mv < cfg["game"]["max_moves"]:
        sims = sims_b if b.to_move == BLACK else sims_w
        temp = 1.0 if mv < temp_moves else 0.0  # randomized openings for variety
        _, acts = search(
            [b], net, device,
            num_simulations=sims, c_puct=m["c_puct"],
            dirichlet_alpha=0.0, dirichlet_frac=0.0,
            komi=komi, temperature=temp, rng=rng,
        )
        b = b.play(acts[0])
        mv += 1
    return b.score(komi)


def crossover(xs, ys, level):
    """Linear-interpolate the x where y first crosses ``level`` (or None)."""
    for i in range(1, len(ys)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 - level) * (y1 - level) <= 0 and y1 != y0:
            t = (level - y0) / (y1 - y0)
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--white-sims", type=int, default=64)
    ap.add_argument("--games", type=int, default=16, help="games per Black-sim level")
    ap.add_argument("--black-grid", default=None,
                    help="comma-separated Black sim counts; default is an auto fine grid")
    ap.add_argument("--out", default="study")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(args.checkpoint, cfg, device)
    temp_moves = cfg["mcts"]["temp_moves"]
    os.makedirs(args.out, exist_ok=True)

    if args.black_grid:
        grid = [int(x) for x in args.black_grid.split(",")]
    else:
        # Fine grid: dense at low counts where the crossover lives, sparser above.
        grid = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
                80, 96, 128]
        grid = [b for b in grid if b <= max(64, args.white_sims * 2)]

    print(f"checkpoint={args.checkpoint}  device={device}  White fixed at {args.white_sims} sims")
    print(f"{args.games} games/level over Black grid: {grid}\n")
    print(f"{'BlackSims':>9} {'BlackWin%':>9} {'meanScore':>10} {'std':>6}")

    rows = []
    for bi, sb in enumerate(grid):
        scores = []
        for g in range(args.games):
            seed = 10_000 * bi + g  # reproducible, distinct per (level, game)
            scores.append(play_game(net, device, cfg, sb, args.white_sims, temp_moves, seed))
        scores = np.array(scores)
        winrate = float((scores > 0).mean())
        rows.append((sb, winrate, float(scores.mean()), float(scores.std())))
        print(f"{sb:>9} {winrate*100:>8.1f}% {scores.mean():>+10.2f} {scores.std():>6.2f}")

    # Save CSV.
    csv_path = os.path.join(args.out, f"handicap_white{args.white_sims}.csv")
    with open(csv_path, "w") as f:
        f.write("black_sims,black_winrate,mean_score,std_score\n")
        for sb, wr, ms, sd in rows:
            f.write(f"{sb},{wr:.4f},{ms:.4f},{sd:.4f}\n")

    xs = [r[0] for r in rows]
    wr = [r[1] for r in rows]
    ms = [r[2] for r in rows]
    sd = [r[3] for r in rows]
    x_cross = crossover(xs, wr, 0.5)
    s_cross = crossover(xs, ms, 0.0)

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(xs, np.array(wr) * 100, "o-", color="C0")
    ax1.axhline(50, ls="--", color="gray", lw=1)
    ax1.axvline(args.white_sims, ls=":", color="C3", lw=1, label=f"equal sims ({args.white_sims})")
    if x_cross is not None:
        ax1.axvline(x_cross, ls="-", color="C2", lw=1.2, label=f"50% crossover ≈ {x_cross:.1f}")
    ax1.set_ylabel("Black win-rate (%)")
    ax1.set_ylim(-3, 103)
    ax1.set_title(f"Handicap study: White fixed at {args.white_sims} sims/move, "
                  f"{args.games} games/level\n(same checkpoint, komi {cfg['game']['komi']})")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.errorbar(xs, ms, yerr=sd, fmt="s-", color="C1", capsize=2)
    ax2.axhline(0, ls="--", color="gray", lw=1)
    if s_cross is not None:
        ax2.axvline(s_cross, ls="-", color="C2", lw=1.2, label=f"even-score ≈ {s_cross:.1f}")
        ax2.legend(loc="best", fontsize=8)
    ax2.set_xlabel("Black sims/move (log scale)")
    ax2.set_ylabel("mean score (Black persp.)")
    ax2.set_xscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(x) for x in xs], fontsize=7, rotation=45)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(args.out, f"handicap_white{args.white_sims}.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")
    if x_cross is not None:
        print(f"\n>>> Black needs ~{x_cross:.0f} sims to break even vs White's {args.white_sims} "
              f"(below that, White wins the majority).")


if __name__ == "__main__":
    main()
