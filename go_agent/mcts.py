"""Batched Monte-Carlo Tree Search (AlphaGo Zero style).

Key points:
  * PUCT selection: ``Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))``.
  * Root gets Dirichlet noise mixed into the priors for exploration.
  * Value is from the perspective of the player to move at a node; it flips
    sign on every step up the tree during backpropagation.
  * Terminal nodes use the true game result (+/-1) instead of a network eval.
  * Batching: one simulation step is run for *all* active games at once, so a
    single GPU forward pass evaluates up to ``len(boards)`` leaves. This is the
    main throughput lever on the 16GB card.

No virtual loss is needed: within a game the simulations are serialized by the
outer loop (sim 0, sim 1, ...), and across games they are batched together.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .encoding import encode_batch
from .game import Board, BLACK, PASS


class Node:
    __slots__ = (
        "board",
        "parent",
        "action",
        "children",
        "N",
        "W",
        "P",
        "expanded",
        "terminal_value",
    )

    def __init__(self, board: Board, parent=None, action=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.action = action
        self.children: dict[int, Node] = {}
        self.N = 0
        self.W = 0.0  # accumulated value, stored from the *parent's* perspective
        self.P = prior
        self.expanded = False
        self.terminal_value = None  # float from this node's to_move perspective, or None

    def select_child(self, c_puct: float):
        total = self.N
        sqrt_total = math.sqrt(total + 1.0)
        best = None
        best_score = -float("inf")
        for a, ch in self.children.items():
            q = ch.W / ch.N if ch.N > 0 else 0.0
            u = c_puct * ch.P * sqrt_total / (1.0 + ch.N)
            s = q + u
            if s > best_score:
                best_score = s
                best = ch
        return best

    def select_leaf(self, c_puct: float):
        node = self
        while node.expanded and node.terminal_value is None:
            node = node.select_child(c_puct)
        return node

    def backprop(self, value: float):
        """Backpropagate ``value`` (from this node's to_move perspective)."""
        cur = value
        node = self
        while node.parent is not None:
            node.W += -cur  # store from parent's perspective
            node.N += 1
            cur = -cur
            node = node.parent
        node.N += 1  # root

    def visit_distribution(self, temperature: float, action_size: int):
        """Return a length-``action_size`` vector of visit probabilities.

        With ``temperature == 0`` this is a one-hot at the most-visited action.
        """
        pi = np.zeros(action_size, dtype=np.float32)
        if not self.children:
            return pi
        if temperature == 0.0:
            best_a = max(self.children, key=lambda a: self.children[a].N)
            pi[best_a] = 1.0
            return pi
        visits = np.array([self.children[a].N for a in self.children], dtype=np.float32)
        temps = visits ** (1.0 / temperature)
        total = temps.sum()
        if total <= 0:
            # Fallback to uniform over visited children.
            for a in self.children:
                pi[a] = 1.0 / len(self.children)
            return pi
        for a, t in zip(self.children, temps):
            pi[a] = t / total
        return pi


def _terminal_value(board: Board, komi: float) -> float:
    """+1 if the player to move has won, -1 if lost (komi avoids draws)."""
    score = board.score(komi)  # black perspective
    winner = BLACK if score > 0 else 2  # 2 == WHITE
    return 1.0 if board.to_move == winner else -1.0


def _masked_policy(logits: np.ndarray, legal: list[int]) -> np.ndarray:
    """Softmax over only the legal actions; returns a full-length vector."""
    probs = np.zeros_like(logits)
    idx = np.array(legal, dtype=np.int64)
    masked = logits[idx]
    masked = masked - masked.max()
    exp = np.exp(masked)
    probs[idx] = exp / exp.sum()
    return probs


def search(
    boards: list[Board],
    net,
    device,
    num_simulations: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    komi: float,
    temperature: float,
    rng: np.random.Generator,
):
    """Run batched MCTS from each board in ``boards``.

    Returns ``(pis, actions)`` where ``pis`` are length-(N*N+1) visit-count
    distributions and ``actions`` are the sampled moves to play.
    """
    N = boards[0].size
    action_size = N * N + 1
    roots = [Node(b.clone()) for b in boards]

    for _ in range(num_simulations):
        leaves: list[Node] = []
        for root in roots:
            if root.terminal_value is not None:
                leaves.append(root)  # already terminal (rare: root is game-over)
                continue
            leaves.append(root.select_leaf(c_puct))

        # Split leaves into terminal (no eval) vs. needs-network.
        net_leaves: list[Node] = []
        for leaf in leaves:
            if leaf.terminal_value is not None:
                leaf.backprop(leaf.terminal_value)
            elif leaf.board.is_terminal():
                leaf.terminal_value = _terminal_value(leaf.board, komi)
                leaf.backprop(leaf.terminal_value)
            else:
                net_leaves.append(leaf)

        if net_leaves:
            batch = encode_batch([lf.board for lf in net_leaves])
            x = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                policy_logits, values = net(x)
            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy()
            for leaf, logits, v in zip(net_leaves, policy_logits, values):
                legal = leaf.board.legal_actions()
                priors = _masked_policy(logits, legal)
                # Add Dirichlet noise at the root only.
                if leaf.parent is None:
                    noise = rng.dirichlet([dirichlet_alpha] * len(legal))
                    for k, a in enumerate(legal):
                        priors[a] = (1 - dirichlet_frac) * priors[a] + dirichlet_frac * noise[k]
                for a in legal:
                    cb = leaf.board.play(a)
                    child = Node(cb, parent=leaf, action=a, prior=float(priors[a]))
                    if cb.is_terminal():
                        child.terminal_value = _terminal_value(cb, komi)
                    leaf.children[a] = child
                leaf.expanded = True
                leaf.backprop(float(v))

    pis = [r.visit_distribution(temperature, action_size) for r in roots]
    actions = [_sample(pi, rng) for pi in pis]
    return pis, actions


def _sample(pi: np.ndarray, rng: np.random.Generator) -> int:
    total = pi.sum()
    if total <= 0:
        return int(rng.integers(0, len(pi)))
    return int(rng.choice(len(pi), p=pi / total))
