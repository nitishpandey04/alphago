"""Play a game against the trained agent from the terminal.

Run with:  uv run python -m scripts.play_human --checkpoint checkpoints/best.pt
Moves are entered GTP-style, e.g. ``A1``, ``C3`` or ``pass``. The agent uses
MCTS (greedy) for its moves.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from go_agent.config import load_config
from go_agent.game import Board, BLACK, WHITE, PASS
from go_agent.model import PolicyValueNet
from go_agent.mcts import search
from go_agent.render import ascii_board, action_to_gtp, save_png

_COLS = "ABCDEFGHJKLMNOPQRST"


def parse_move(text: str, size: int):
    text = text.strip().lower()
    if text in ("pass", "p"):
        return size * size
    if len(text) < 2:
        return None
    col = text[0].upper()
    row = text[1:]
    if col not in _COLS[:size] or not row.isdigit():
        return None
    c = _COLS.index(col)
    r = size - int(row)
    if not (0 <= r < size and 0 <= c < size):
        return None
    return r * size + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--human-color", default="black", choices=["black", "white"])
    ap.add_argument("--sims", type=int, default=None, help="override MCTS simulations")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg["game"]["board_size"]
    net = PolicyValueNet(
        board_size=size,
        in_channels=cfg["model"]["in_channels"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        channels=cfg["model"]["channels"],
        value_channels=cfg["model"]["value_channels"],
        hidden_size=cfg["model"]["hidden_size"],
    ).to(device)
    if os.path.exists(args.checkpoint):
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"WARNING: {args.checkpoint} not found; using a random (untrained) net")
    net.eval()

    human_color = BLACK if args.human_color == "black" else WHITE
    rng = np.random.default_rng(0)
    sims = args.sims or cfg["mcts"]["num_simulations"]
    board = Board(size)

    print(f"You are {'Black (X)' if human_color == BLACK else 'White (O)'}.")
    print("Enter moves like A1, B3, or 'pass'. Ctrl-C to quit.\n")
    print(ascii_board(board))

    while not board.is_terminal():
        if board.to_move == human_color:
            while True:
                try:
                    mv = input("your move > ")
                except (EOFError, KeyboardInterrupt):
                    print("\nbye"); return
                a = parse_move(mv, size)
                if a is None or not board.is_legal(a):
                    print("illegal move, try again"); continue
                break
            board = board.play(a)
            print(f"you played {action_to_gtp(a, size)}")
        else:
            print("agent thinking...")
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
            print(f"agent played {action_to_gtp(a, size)}")
        print(ascii_board(board, last_move=board.last_move if board.last_move is not PASS else None))
        print()

    score = board.score(cfg["game"]["komi"])
    winner = "Black" if score > 0 else "White"
    print(f"Game over. Winner: {winner} (score={score:+.1f})")


if __name__ == "__main__":
    main()
