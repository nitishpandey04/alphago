"""Policy-value network (AlphaGo Zero architecture, small).

Trunk: initial 3x3 conv -> residual blocks. Two heads:
  * policy head -> logits over N*N+1 actions (incl. pass)
  * value head  -> scalar in [-1, 1] (expected result for the side to move)

Fully convolutional except for a single linear layer at the tip of each head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        board_size: int,
        in_channels: int = 4,
        num_res_blocks: int = 4,
        channels: int = 64,
        value_channels: int = 8,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1  # +1 for pass

        self.trunk = nn.Sequential(
            ConvBlock(in_channels, channels),
            *[ResBlock(channels) for _ in range(num_res_blocks)],
        )

        # Policy head: 1x1 conv to 2 channels -> flatten -> linear to actions.
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head: 1x1 conv -> flatten -> hidden -> scalar -> tanh.
        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_size * board_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.trunk(x)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)
