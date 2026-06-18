"""Batched self-play with a rolling pool of concurrent games.

A fixed number of games run in parallel. Each "step" advances every active game
by one move using a single batched MCTS call (so all leaf evaluations share one
GPU forward pass). When a game finishes its samples are collected and a fresh
game is started in that slot, until ``target_games`` total games are produced.
"""

from __future__ import annotations

import numpy as np

from .encoding import encode
from .game import Board, BLACK
from .mcts import search


class _Game:
    __slots__ = ("board", "move_num", "states", "pis", "legals", "players", "done")

    def __init__(self, size: int):
        self.board = Board(size)
        self.move_num = 0
        self.states = []
        self.pis = []
        self.legals = []
        self.players = []
        self.done = False

    def record(self, state, pi, legal_mask):
        self.states.append(state)
        self.pis.append(pi)
        self.legals.append(legal_mask)
        self.players.append(self.board.to_move)

    def finalize(self, komi: float):
        score = self.board.score(komi)  # black perspective
        winner = BLACK if score > 0 else 2
        zs = np.array([1.0 if p == winner else -1.0 for p in self.players], dtype=np.float32)
        states = np.stack(self.states).astype(np.float32)
        pis = np.stack(self.pis).astype(np.float32)
        legals = np.stack(self.legals).astype(np.bool_)
        self.done = True
        return states, pis, legals, zs


def generate_games(
    net,
    device,
    cfg,
    rng: np.random.Generator,
    target_games: int,
    num_parallel_games: int,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    temp_moves: int,
    komi: float,
    max_moves: int,
):
    """Run self-play and return a list of (states, pis, legals, zs) per game."""
    size = cfg["game"]["board_size"]
    action_size = size * size + 1

    pool = [_Game(size) for _ in range(min(num_parallel_games, target_games))]
    started = len(pool)
    results = []

    while len(results) < target_games:
        active = [g for g in pool if not g.done]
        if not active:
            break
        boards = [g.board for g in active]
        # Temperature: 1 early in each game, 0 (greedy) after temp_moves.
        temps = [1.0 if g.move_num < temp_moves else 0.0 for g in active]
        # search() uses a single temperature; run groups by temperature value.
        # In practice split into two batched searches.
        pis_all = [None] * len(active)
        actions_all = [None] * len(active)
        for tval in (1.0, 0.0):
            sel = [i for i, g in enumerate(active) if temps[i] == tval]
            if not sel:
                continue
            sub_boards = [boards[i] for i in sel]
            pis, acts = search(
                sub_boards,
                net,
                device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_frac=dirichlet_frac,
                komi=komi,
                temperature=tval,
                rng=rng,
            )
            for j, i in enumerate(sel):
                pis_all[i] = pis[j]
                actions_all[i] = acts[j]

        for g, pi, a in zip(active, pis_all, actions_all):
            legal_mask = np.zeros(action_size, dtype=np.bool_)
            legal_mask[g.board.legal_actions()] = True
            g.record(encode(g.board), pi, legal_mask)
            g.board = g.board.play(a)
            g.move_num += 1
            if g.board.is_terminal() or g.move_num >= max_moves:
                results.append(g.finalize(komi))

        # Refill finished slots with fresh games (rolling pool).
        new_pool = []
        for g in pool:
            if g.done:
                if started < target_games and len(results) < target_games:
                    new_pool.append(_Game(size))
                    started += 1
            else:
                new_pool.append(g)
        pool = new_pool

    return results[:target_games]
