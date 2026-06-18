"""Go game rules: board, legal moves, simple ko, area scoring.

Board stones: 0=empty, 1=black, 2=white. Black moves first.
Action space: indices 0..N*N-1 map to (row, col) via ``a = row * N + col``;
the final index ``N*N`` is the PASS action.

Note on rules: we implement *simple ko* (no positional superko) and area
scoring (stones + surrounded empty territory) with komi for White. This keeps
the ruleset tractable while staying close to standard Go.
"""

from __future__ import annotations

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

PASS = -1  # internal sentinel for a pass move; action index is N*N


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class Board:
    """A Go board with group/liberty tracking via flood fill.

    The board is mutable via :meth:`play`. ``play`` returns a *new* Board so
    that MCTS can branch without deep-copying history beyond what's needed.
    """

    __slots__ = (
        "size",
        "grid",
        "to_move",
        "ko_point",
        "last_move",
        "passes",
        "move_count",
    )

    def __init__(self, size: int):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.to_move = BLACK
        self.ko_point = None  # (row, col) forbidden for the player to move, or None
        self.last_move = None  # (row, col) or PASS
        self.passes = 0  # consecutive passes
        self.move_count = 0

    # -- introspection ---------------------------------------------------
    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.size = self.size
        b.grid = self.grid.copy()
        b.to_move = self.to_move
        b.ko_point = self.ko_point
        b.last_move = self.last_move
        b.passes = self.passes
        b.move_count = self.move_count
        return b

    def is_terminal(self) -> bool:
        return self.passes >= 2

    def neighbors(self, r: int, c: int):
        if r > 0:
            yield r - 1, c
        if r < self.size - 1:
            yield r + 1, c
        if c > 0:
            yield r, c - 1
        if c < self.size - 1:
            yield r, c + 1

    def _group_and_liberties(self, r: int, c: int):
        """Flood-fill the group at (r,c); return (stones set, liberties set)."""
        color = int(self.grid[r, c])
        if color == EMPTY:
            return set(), set()
        stones: set[tuple[int, int]] = set()
        libs: set[tuple[int, int]] = set()
        stack = [(r, c)]
        stones.add((r, c))
        while stack:
            cr, cc = stack.pop()
            for nr, nc in self.neighbors(cr, cc):
                v = int(self.grid[nr, nc])
                if v == EMPTY:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in stones:
                    stones.add((nr, nc))
                    stack.append((nr, nc))
        return stones, libs

    # -- legality & play -------------------------------------------------
    def is_legal(self, action: int) -> bool:
        """``action`` in [0, N*N]; ``N*N`` is pass and is always legal."""
        if action == self.size * self.size:
            return True
        r, c = divmod(action, self.size)
        if int(self.grid[r, c]) != EMPTY:
            return False
        if self.ko_point is not None and (r, c) == self.ko_point:
            return False
        # Simulate to check for suicide.
        ok, _ = self._simulate_place(r, c, self.to_move)
        return ok

    def legal_actions(self) -> list[int]:
        """All legal actions including the pass action (last index)."""
        acts = []
        N = self.size
        for a in range(N * N):
            if self.is_legal(a):
                acts.append(a)
        acts.append(N * N)  # pass
        return acts

    def _simulate_place(self, r: int, c: int, color: int):
        """Check whether placing ``color`` at (r,c) is non-suicide.

        Returns (legal, captured_stones). Does not mutate this board.
        """
        opp = opponent(color)
        # Tentatively place.
        self.grid[r, c] = color
        captured: list[tuple[int, int]] = []
        for nr, nc in self.neighbors(r, c):
            if int(self.grid[nr, nc]) == opp:
                stones, libs = self._group_and_liberties(nr, nc)
                if not libs:
                    captured.extend(stones)
        # Remove captured (tentatively) to test own liberties.
        for (cr, cc) in captured:
            self.grid[cr, cc] = EMPTY
        _, own_libs = self._group_and_liberties(r, c)
        legal = len(own_libs) > 0
        # Undo tentative changes.
        for (cr, cc) in captured:
            self.grid[cr, cc] = opp
        self.grid[r, c] = EMPTY
        return legal, captured

    def play(self, action: int) -> "Board":
        """Return a new Board after playing ``action`` for the player to move."""
        N = self.size
        b = self.clone()
        color = b.to_move

        if action == N * N:  # pass
            b.last_move = PASS
            b.passes = b.passes + 1
            b.ko_point = None
            b.to_move = opponent(color)
            b.move_count = b.move_count + 1
            return b

        r, c = divmod(action, N)
        if int(b.grid[r, c]) != EMPTY:
            raise ValueError(f"illegal move: {action} ({r},{c}) occupied")
        if b.ko_point is not None and (r, c) == b.ko_point:
            raise ValueError(f"illegal move: {action} ({r},{c}) is a ko point")

        opp = opponent(color)
        b.grid[r, c] = color
        captured: list[tuple[int, int]] = []
        for nr, nc in b.neighbors(r, c):
            if int(b.grid[nr, nc]) == opp:
                stones, libs = b._group_and_liberties(nr, nc)
                if not libs:
                    captured.extend(stones)
        for (cr, cc) in captured:
            b.grid[cr, cc] = EMPTY

        _, own_libs = b._group_and_liberties(r, c)
        if not own_libs:
            raise ValueError(f"illegal move: {action} ({r},{c}) is suicide")

        # Simple ko: capturing exactly one stone with a single-stone group
        # that has exactly one liberty (the captured point).
        new_ko = None
        if len(captured) == 1:
            own_stones, own_libs2 = b._group_and_liberties(r, c)
            if len(own_stones) == 1 and len(own_libs2) == 1:
                new_ko = captured[0]

        b.ko_point = new_ko
        b.last_move = (r, c)
        b.passes = 0
        b.to_move = opp
        b.move_count = b.move_count + 1
        return b

    # -- scoring ---------------------------------------------------------
    def score(self, komi: float) -> float:
        """Area score from Black's perspective: black_area - (white_area + komi).

        Positive => Black wins. Call only on terminal positions.
        """
        N = self.size
        black = int(np.count_nonzero(self.grid == BLACK))
        white = int(np.count_nonzero(self.grid == WHITE))
        visited = np.zeros((N, N), dtype=bool)
        for r in range(N):
            for c in range(N):
                if int(self.grid[r, c]) != EMPTY or visited[r, c]:
                    continue
                # Flood empty region; track bordering colors.
                region = []
                borders = set()
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    region.append((cr, cc))
                    for nr, nc in self.neighbors(cr, cc):
                        v = int(self.grid[nr, nc])
                        if v == EMPTY:
                            if not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                        else:
                            borders.add(v)
                if borders == {BLACK}:
                    black += len(region)
                elif borders == {WHITE}:
                    white += len(region)
        return float(black - (white + komi))

    # -- convenience for agents / rendering ------------------------------
    def as_planes_for(self, color: int) -> np.ndarray:
        """Return (2, N, N) planes: [my stones, opponent stones] for ``color``."""
        mine = (self.grid == color).astype(np.int8)
        opp = (self.grid == opponent(color)).astype(np.int8)
        return np.stack([mine, opp], axis=0)

    def __repr__(self) -> str:
        rows = []
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        for r in range(self.size):
            rows.append(" ".join(sym[int(v)] for v in self.grid[r]))
        who = "B" if self.to_move == BLACK else "W"
        return "\n".join(rows) + f"\n(to move: {who})"
