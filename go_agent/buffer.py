"""Replay buffer of self-play samples (state, pi, legal_mask, z)."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape):
        self.capacity = capacity
        self.state_shape = tuple(state_shape)
        self.states = np.empty((capacity,) + self.state_shape, dtype=np.float32)
        self.pis = None       # allocated on first add (action_size known then)
        self.legal = None
        self.values = np.empty((capacity,), dtype=np.float32)
        self.action_size = None
        self.size = 0
        self.idx = 0

    def add(self, states: np.ndarray, pis: np.ndarray, legal: np.ndarray, zs: np.ndarray):
        if self.action_size is None:
            self.action_size = pis.shape[1]
            self.pis = np.empty((self.capacity, self.action_size), dtype=np.float32)
            self.legal = np.empty((self.capacity, self.action_size), dtype=np.bool_)
        n = states.shape[0]
        for i in range(n):
            j = self.idx
            self.states[j] = states[i]
            self.pis[j] = pis[i]
            self.legal[j] = legal[i]
            self.values[j] = zs[i]
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        if self.size == 0:
            raise ValueError("buffer is empty")
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.pis[idx],
            self.legal[idx],
            self.values[idx],
        )

    def __len__(self):
        return self.size
