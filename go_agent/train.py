"""Training: policy cross-entropy + value MSE (+ L2 via weight decay).

The policy target ``pi`` is a distribution over legal moves (zero elsewhere).
Illegal-move logits are masked to -inf before log-softmax so the network's
probability mass is concentrated on legal actions, matching the target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, net, lr: float, l2_weight: float, policy_weight: float, value_weight: float, device):
        self.net = net
        self.device = device
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=l2_weight)

    def train_step(self, states, pis, legals, zs):
        x = torch.from_numpy(states).to(self.device)
        pi_t = torch.from_numpy(pis).to(self.device)
        legal_t = torch.from_numpy(legals).to(self.device)
        z_t = torch.from_numpy(zs).to(self.device)

        logits, value = self.net(x)

        # Mask illegal moves before computing the policy distribution.
        neg_inf = torch.full_like(logits, float("-inf"))
        masked_logits = torch.where(legal_t, logits, neg_inf)
        log_probs = F.log_softmax(masked_logits, dim=1)
        # Zero out illegal entries (log_probs there are -inf) so the 0*-inf
        # product with the zero target doesn't produce NaN.
        log_probs = torch.where(legal_t, log_probs, torch.zeros_like(log_probs))
        policy_loss = -(pi_t * log_probs).sum(dim=1).mean()

        value_loss = F.mse_loss(value, z_t)

        loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }
