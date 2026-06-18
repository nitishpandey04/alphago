"""Board -> neural network input planes.

Encoding (from the perspective of the player to move):
  plane 0: current player's stones
  plane 1: opponent's stones
  plane 2: ko point (1 at the forbidden recapture, else 0)
  plane 3: color-to-move flag (1 if White to move, 0 if Black)

This matches ``model.in_channels`` in the config (4 planes). Encoding is always
relative to the side to move so the network learns a single perspective.
"""

from __future__ import annotations

import numpy as np

from .game import Board, EMPTY, opponent

NUM_PLANES = 4


def encode(board: Board) -> np.ndarray:
    """Return a (NUM_PLANES, N, N) float32 array for ``board``."""
    N = board.size
    to_move = board.to_move
    mine = (board.grid == to_move).astype(np.float32)
    opp = (board.grid == opponent(to_move)).astype(np.float32)

    ko = np.zeros((N, N), dtype=np.float32)
    if board.ko_point is not None:
        kr, kc = board.ko_point
        ko[kr, kc] = 1.0

    color = np.full((N, N), 1.0 if to_move == 2 else 0.0, dtype=np.float32)

    return np.stack([mine, opp, ko, color], axis=0)


def encode_batch(boards: list[Board]) -> np.ndarray:
    """Return a (B, NUM_PLANES, N, N) float32 array."""
    arr = np.empty((len(boards), NUM_PLANES, boards[0].size, boards[0].size), dtype=np.float32)
    for i, b in enumerate(boards):
        arr[i] = encode(b)
    return arr
