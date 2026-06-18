"""Watch a self-play or evaluation game play out move-by-move in the terminal.

Run with:  uv run python -m scripts.watch_game --checkpoint checkpoints/best.pt
Optionally pit two checkpoints with --black and --white.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from go_agent.config import load_config
from go_agent.game import Board, BLACK, PASS
from go_agent.model import PolicyValueNet
from go_agent.mcts import search
from go_agent.render import ascii_board, action_to_gtp


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--black", default="checkpoints/best.pt")
    ap.add_argument("--white", default="checkpoints/best.pt")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--save-pngs", default=None, help="directory to save per-move PNGs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["game"]["board_size"]
    net_b = load_net(args.black, cfg, device)
    net_w = load_net(args.white, cfg, device)
    rng = np.random.default_rng(0)
    sims = args.sims or cfg["mcts"]["num_simulations"]

    if args.save_pngs:
        os.makedirs(args.save_pngs, exist_ok=True)
        from go_agent.render import save_png

    board = Board(size)
    move = 0
    print(ascii_board(board))
    while not board.is_terminal() and move < cfg["game"]["max_moves"]:
        net = net_b if board.to_move == BLACK else net_w
        _, actions = search(
            [board], net, device,
            num_simulations=sims,
            c_puct=cfg["mcts"]["c_puct"],
            dirichlet_alpha=0.0,
            dirichlet_frac=0.0,
            komi=cfg["game"]["komi"],
            temperature=0.0,
            rng=rng,
        )
        a = actions[0]
        board = board.play(a)
        move += 1
        last = board.last_move if board.last_move is not PASS else None
        print(f"\n--- move {move}: {action_to_gtp(a, size)} ---")
        print(ascii_board(board, last_move=last))
        if args.save_pngs:
            save_png(board, os.path.join(args.save_pngs, f"move_{move:03d}.png"), last_move=last)

    score = board.score(cfg["game"]["komi"])
    print(f"\nGame over. Winner: {'Black' if score > 0 else 'White'} (score={score:+.1f})")


if __name__ == "__main__":
    main()
