"""Main training loop: self-play -> train -> arena -> checkpoint, repeated.

Run with:  uv run python -m scripts.train_main --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import numpy as np
import torch

from go_agent.arena import evaluate
from go_agent.buffer import ReplayBuffer
from go_agent.config import load_config
from go_agent.logging_utils import Logger
from go_agent.model import PolicyValueNet
from go_agent.selfplay import generate_games
from go_agent.train import Trainer


def build_net(cfg, device):
    m = cfg["model"]
    return PolicyValueNet(
        board_size=cfg["game"]["board_size"],
        in_channels=m["in_channels"],
        num_res_blocks=m["num_res_blocks"],
        channels=m["channels"],
        value_channels=m["value_channels"],
        hidden_size=m["hidden_size"],
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config)
    r = cfg["runtime"]
    device = torch.device(r["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    rng = np.random.default_rng(r["seed"])
    torch.manual_seed(r["seed"])
    np.random.seed(r["seed"])

    os.makedirs(r["checkpoint_dir"], exist_ok=True)
    logger = Logger(r["log_dir"])

    net = build_net(cfg, device)
    best_net = build_net(cfg, device)
    if args.resume and os.path.exists(args.resume):
        best_net.load_state_dict(torch.load(args.resume, map_location=device))
        net.load_state_dict(best_net.state_dict())
        print(f"Resumed from {args.resume}")
    best_net.eval()
    net.eval()

    trainer = Trainer(
        net,
        lr=cfg["train"]["lr"],
        l2_weight=cfg["train"]["l2_weight"],
        policy_weight=cfg["train"]["policy_weight"],
        value_weight=cfg["train"]["value_weight"],
        device=device,
    )

    size = cfg["game"]["board_size"]
    in_ch = cfg["model"]["in_channels"]
    buffer = ReplayBuffer(cfg["train"]["buffer_size"], (in_ch, size, size))

    sp = cfg["selfplay"]
    m = cfg["mcts"]
    tr = cfg["train"]

    for it in range(cfg["loop"]["num_iterations"]):
        # ---- Self-play with the best net ----
        best_net.eval()
        games = generate_games(
            net=best_net,
            device=device,
            cfg=cfg,
            rng=rng,
            target_games=sp["target_games"],
            num_parallel_games=sp["num_parallel_games"],
            num_simulations=m["num_simulations"],
            c_puct=m["c_puct"],
            dirichlet_alpha=m["dirichlet_alpha"],
            dirichlet_frac=m["dirichlet_frac"],
            temp_moves=m["temp_moves"],
            komi=cfg["game"]["komi"],
            max_moves=cfg["game"]["max_moves"],
        )
        for states, pis, legals, zs in games:
            buffer.add(states, pis, legals, zs)

        total_positions = sum(s.shape[0] for s, _, _, _ in games)
        print(f"[iter {it}] self-play: {len(games)} games, {total_positions} positions, buffer={len(buffer)}")

        # ---- Train ----
        net.train()
        metrics_accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
        for _ in range(tr["num_steps"]):
            if len(buffer) < tr["batch_size"]:
                break
            states, pis, legals, zs = buffer.sample(tr["batch_size"], rng)
            mt = trainer.train_step(states, pis, legals, zs)
            for k in metrics_accum:
                metrics_accum[k] += mt[k]
        avg = {k: v / max(1, tr["num_steps"]) for k, v in metrics_accum.items()}
        print(f"[iter {it}] train: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        logger.step = it
        logger.log_scalars("train", avg, step=it)
        logger.log_scalar("buffer/size", len(buffer), step=it)
        logger.log_scalar("selfplay/positions", total_positions, step=it)
        net.eval()

        # ---- Arena: candidate (net) vs best ----
        if (it + 1) % cfg["loop"]["eval_every"] == 0:
            win_rate = evaluate(net, best_net, device, cfg, rng)
            print(f"[iter {it}] arena: candidate win-rate vs best = {win_rate:.2f}")
            logger.log_scalar("arena/win_rate_vs_best", win_rate, step=it)
            if win_rate >= cfg["arena"]["win_threshold"]:
                best_net.load_state_dict(net.state_dict())
                print(f"[iter {it}] candidate PROMOTED to best")
                ckpt = os.path.join(r["checkpoint_dir"], "best.pt")
                torch.save(best_net.state_dict(), ckpt)
                print(f"  saved {ckpt}")
            else:
                # Revert net to best so self-play keeps improving from the best.
                net.load_state_dict(best_net.state_dict())

        # Periodic checkpoint regardless.
        if (it + 1) % 5 == 0:
            torch.save(best_net.state_dict(), os.path.join(r["checkpoint_dir"], f"best_iter{it}.pt"))

    torch.save(best_net.state_dict(), os.path.join(r["checkpoint_dir"], "best.pt"))
    logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
