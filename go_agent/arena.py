"""Arena: evaluate a candidate network against the current best.

Each game is played with MCTS (greedy, temperature 0) for both sides using their
respective networks. Colors are alternated to cancel any first-move advantage.
A candidate is promoted only if its win rate exceeds ``win_threshold``.
"""

from __future__ import annotations

import numpy as np

from .game import Board, BLACK
from .mcts import search


def play_match(net_black, net_white, device, cfg, rng, max_moves: int):
    """Play one game; return 1 if the candidate-style 'black net' wins, else 0.

    Here ``net_black`` plays Black and ``net_white`` plays White. The caller
    swaps which network is which to balance colors.
    """
    size = cfg["game"]["board_size"]
    komi = cfg["game"]["komi"]
    m = cfg["mcts"]
    board = Board(size)
    move_num = 0
    while not board.is_terminal() and move_num < max_moves:
        net = net_black if board.to_move == BLACK else net_white
        _, actions = search(
            [board],
            net,
            device,
            num_simulations=m["num_simulations"],
            c_puct=m["c_puct"],
            dirichlet_alpha=0.0,  # no noise in evaluation
            dirichlet_frac=0.0,
            komi=komi,
            temperature=0.0,
            rng=rng,
        )
        board = board.play(actions[0])
        move_num += 1

    score = board.score(komi)  # black perspective
    return 1 if score > 0 else 0


def evaluate(candidate, best, device, cfg, rng):
    """Return the candidate's win rate over ``arena.num_games`` games."""
    n = cfg["arena"]["num_games"]
    max_moves = cfg["game"]["max_moves"]
    wins = 0
    for i in range(n):
        if i % 2 == 0:
            wins += play_match(candidate, best, device, cfg, rng, max_moves)
        else:
            wins += 1 - play_match(best, candidate, device, cfg, rng, max_moves)
    return wins / n
