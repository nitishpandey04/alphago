"""Board rendering: ASCII for the terminal and PNG via matplotlib."""

from __future__ import annotations

import numpy as np

from .game import Board, EMPTY, BLACK, WHITE, PASS

_GLYPH = {EMPTY: ".", BLACK: "X", WHITE: "O"}
_COLS = "ABCDEFGHJKLMNOPQRST"  # skip 'I' (Go convention)


def ascii_board(board: Board, last_move=None) -> str:
    """Return a printable ASCII board with coordinates and last-move marker."""
    N = board.size
    header = "   " + " ".join(_COLS[i] for i in range(N))
    lines = [header]
    for r in range(N):
        row_label = str(N - r)
        cells = []
        for c in range(N):
            v = int(board.grid[r, c])
            if last_move is not None and last_move is not PASS and last_move == (r, c):
                cells.append(_GLYPH[v].lower())  # lowercase = last move
            else:
                cells.append(_GLYPH[v])
        lines.append(f"{row_label:>2} " + " ".join(cells))
    lines.append(header)
    who = "Black (X)" if board.to_move == BLACK else "White (O)"
    if board.is_terminal():
        status = "  [game over]"
    else:
        status = f"  to move: {who}"
    lines.append(status)
    return "\n".join(lines)


def save_png(board: Board, path: str, last_move=None, title: str | None = None):
    """Save a PNG image of the board."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = board.size
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_facecolor("#e8b06c")
    # Grid lines.
    for i in range(N):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
        ax.axvline(i - 0.5, color="black", linewidth=0.5)
    ax.set_xticks(range(N))
    ax.set_xticklabels([_COLS[i] for i in range(N)])
    ax.set_yticks(range(N))
    ax.set_yticklabels([str(N - i) for i in range(N)])
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for r in range(N):
        for c in range(N):
            v = int(board.grid[r, c])
            if v == EMPTY:
                continue
            color = "black" if v == BLACK else "white"
            ax.scatter(c, r, s=420, c=color, edgecolors="black", zorder=3)
            if last_move is not None and last_move is not PASS and last_move == (r, c):
                ax.scatter(c, r, s=80, c="red", zorder=4)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def action_to_gtp(action: int, size: int) -> str:
    """Convert an action index to a human-readable move like 'D4' or 'pass'."""
    if action == size * size:
        return "pass"
    r, c = divmod(action, size)
    return f"{_COLS[c]}{size - r}"
