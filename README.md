# alphago — AlphaGo Zero–style Go agent (self-play, from scratch)

A small, readable, from-scratch implementation of the **AlphaGo Zero** algorithm
that learns to play Go **purely from self-play** — no human games, no
supervised pretraining, no expert knowledge. The agent starts knowing nothing
but the rules and improves itself by playing against its own best version.

It targets an **NVIDIA RTX 5060 Ti (16 GB, Blackwell sm_120)** and uses
PyTorch's CUDA 12.8 wheels. The whole pipeline — self-play, MCTS, training, and
evaluation — runs on the GPU with batched leaf evaluations.

> Start on a 5×5 board to debug the full loop fast, then scale to 7×7 or 9×9 by
> editing a few numbers in a YAML config. No code changes required.

---

## Table of contents

- [Why AlphaGo Zero?](#why-alphago-zero)
- [How it works (the algorithm)](#how-it-works-the-algorithm)
  - [The self-play loop](#the-self-play-loop)
  - [Monte-Carlo Tree Search (MCTS)](#monte-carlo-tree-search-mcts)
  - [The policy-value network](#the-policy-value-network)
  - [Training objective](#training-objective)
  - [The arena (promotion gate)](#the-arena-promotion-gate)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Quick smoke test](#quick-smoke-test)
- [Training](#training)
- [Playing against the agent](#playing-against-the-agent)
- [Watching games](#watching-games)
- [The handicap study: how big is the first-move advantage?](#the-handicap-study-how-big-is-the-first-move-advantage)
- [Configuration reference](#configuration-reference)
- [Scaling to bigger boards](#scaling-to-bigger-boards)
- [Performance & GPU notes](#performance--gpu-notes)
- [Limitations & future work](#limitations--future-work)

---

## Why AlphaGo Zero?

The original AlphaGo had three stages: (1) supervised learning on human games,
(2) reinforcement learning by self-play, (3) a value network + MCTS at
inference. It needed a large dataset of expert games.

**AlphaGo Zero** drops stage 1 entirely. It learns from self-play alone and
fuses the policy and value networks into one. That makes it the right fit when
you have **no game traces** — which is exactly the case here. It's also simpler
and stronger (AlphaGo Zero beat the original AlphaGo 100–0).

This project is a compact, pedagogical reimplementation: small enough to read
in an evening, faithful enough to reproduce the core ideas.

---

## How it works (the algorithm)

### The self-play loop

Each training iteration repeats the same four steps:

```
   ┌──────────────────────────────────────────────────────────────┐
   │  1. SELF-PLAY      best_net plays N games against itself     │
   │                     -> record (state, pi, z) per position    │
   │  2. TRAIN          update candidate_net on the replay buffer │
   │                     loss = policy_CE + value_MSE + L2        │
   │  3. ARENA          candidate_net vs best_net, colors swapped │
   │                     win_rate > 55% ? promote : revert        │
   │  4. REPEAT         with the (possibly new) best_net          │
   └──────────────────────────────────────────────────────────────┘
```

Self-play uses the **current best network** for both sides. Every position
encountered is stored as a training sample:

| field | meaning |
|-------|---------|
| `state`  | the board encoded as 4 input planes (see [network](#the-policy-value-network)) |
| `pi`     | the MCTS visit-count distribution — *the improved policy target* |
| `legal`  | a boolean mask of legal moves (used to mask the policy loss) |
| `z`      | the final game result, `+1` if the side to move won, `-1` if it lost |

The trick that makes this work: MCTS produces a **stronger policy than the raw
network** (it's the network's prior refined by search). So the network is
trained to *predict its own search-improved policy*, bootstrapping itself
upward with no external signal.

### Monte-Carlo Tree Search (MCTS)

For each move we run `num_simulations` (default 64) MCTS rollouts. Each rollout:

1. **Selection** — from the root, descend by the PUCT formula until reaching a
   leaf:
   ```
   a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a)) ]
   ```
   `Q` is the mean value of action `a`, `P` is the network's prior, `N` are
   visit counts. The exploration term favors high-prior, under-explored moves.
2. **Evaluation** — at the leaf, ask the network for `(policy_logits, value)`.
   The policy logits are softmaxed over **legal moves only** to get priors; the
   value is the network's estimate of the leaf's worth for the side to move.
   Terminal leaves (game over) use the true game result instead of the network.
3. **Expansion** — add a child node for every legal move, with the network
   priors.
4. **Backpropagation** — walk back to the root, flipping the sign of the value
   at each level (because players alternate). Accumulate `W` and increment `N`.

After all simulations, the **root's visit counts become the policy target**.
With temperature `τ=1` early in the game (exploration) and `τ=0` later (greedy
on the most-visited move), following AlphaGo Zero's schedule.

**Dirichlet root noise:** at the root we mix the priors with Dirichlet noise
(`α≈0.03, fraction=0.25`) to ensure diverse move choices across self-play
games. This is why the agent doesn't collapse to always playing the same
opening.

**Batching (the GPU win):** instead of running one simulation per game at a
time, we maintain a pool of `num_parallel_games` (default 128) concurrent
games. On each MCTS step we gather *every pending leaf across all games* and
evaluate them in **one batched GPU forward pass**, then scatter the results
back. This is the 10–30× throughput lever on the GPU — the network sees a big
batch instead of one board at a time.

### The policy-value network

A single convolutional network with two heads (the AlphaGo Zero architecture,
scaled down):

```
input (4, N, N)              # own stones, opp stones, ko point, color-to-move
   │
   ▼
┌─────────────────┐
│ 3×3 conv + BN   │  ─┐
│ + ReLU          │   │  × num_res_blocks (default 4)
└─────────────────┘  ─┘   residual blocks
   │
   ├─── POLICY HEAD ─── 1×1 conv (2 ch) → flatten → Linear(N*N*2, N*N+1)
   │                     logits over all moves incl. PASS
   │
   └─── VALUE HEAD  ─── 1×1 conv (8 ch) → flatten → Linear → ReLU → Linear(1) → tanh
                         scalar in [-1, +1], the expected result for the side to move
```

Why these design choices:
- **4 input planes** instead of the historical 48-feature encoding — enough to
  represent the state compactly while staying small. The encoding is always
  from the **side-to-move's perspective** so the network learns one viewpoint.
- **Residual blocks** let the trunk deepen without degradation.
- **BatchNorm** stabilizes training on the small batches self-play produces.
- One shared trunk → two heads is more sample-efficient than separate nets and
  matches the AlphaGo Zero design.

At 5×5 the network is ~310k parameters — tiny, so most of the runtime is
actually in the MCTS tree traversal (see [performance notes](#performance--gpu-notes)).

### Training objective

```
L = policy_weight · CE(softmax(logits | legal), pi)
  + value_weight · MSE(value, z)
  + L2  (via Adam weight_decay)
```

The policy cross-entropy is computed over **legal moves only**: illegal logits
are masked to `-inf` before log-softmax, and their log-probs are zeroed so the
`0 · (-inf) = NaN` doesn't appear. The target `pi` is the MCTS visit
distribution, which is zero on illegal moves by construction.

### The arena (promotion gate)

After training, the candidate network plays `arena.num_games` games against the
current best, alternating colors to cancel first-move advantage. If the
candidate's win rate exceeds `win_threshold` (default 0.55) it is **promoted**
to best and saved to `checkpoints/best.pt`. Otherwise the candidate is
discarded and self-play continues from the old best — this prevents the network
from regressing.

---

## Project layout

```
alphago/
├── pyproject.toml            # uv project; pins torch from the cu128 index
├── configs/
│   ├── default.yaml          # main config: 5×5, 128 parallel games, 64 sims
│   └── smoke.yaml            # tiny config for a ~1-minute pipeline check
├── go_agent/
│   ├── game.py               # Go rules: board, captures, simple ko, area scoring
│   ├── encoding.py           # board → 4 NN input planes (side-to-move relative)
│   ├── model.py              # residual policy-value network
│   ├── mcts.py               # batched PUCT MCTS with Dirichlet root noise
│   ├── selfplay.py           # rolling pool of batched self-play games
│   ├── buffer.py             # replay buffer (state, pi, legal, z)
│   ├── train.py              # masked policy CE + value MSE + L2 step
│   ├── arena.py              # candidate-vs-best evaluation with color balancing
│   ├── logging_utils.py      # thin TensorBoard wrapper
│   └── render.py             # ASCII + matplotlib board rendering
└── scripts/
    ├── train_main.py         # the training loop (self-play → train → arena → ckpt)
    ├── play_human.py         # play against the agent in the terminal
    ├── watch_game.py         # watch a self-play / eval game move-by-move
    └── handicap_study.py     # sweep per-side MCTS sims; plot win-rate vs handicap
```

---

## Setup

Requires an NVIDIA GPU. The RTX 50-series (Blackwell, compute capability 12.0)
needs PyTorch's **CUDA 12.8** wheels, which are already pinned in
`pyproject.toml` via the `pytorch-cu128` index. Uses [uv](https://docs.astral.sh/uv/)
for dependency management.

```bash
cd alphago
uv sync          # installs torch 2.11+cu128, numpy, tensorboard, matplotlib, etc.
```

Verify the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
# -> NVIDIA GeForce RTX 5060 Ti
```

(On CPU-only machines it falls back automatically, but it will be very slow.)

---

## Quick smoke test

Runs the full pipeline — self-play → train → arena → checkpoint — in about a
minute on a tiny config (16 games, 32 sims, 2 iterations):

```bash
uv run python -m scripts.train_main --config configs/smoke.yaml
```

You should see something like:

```
[iter 0] self-play: 16 games, 537 positions, buffer=537
[iter 0] train: loss=2.2971, policy_loss=1.9968, value_loss=0.3003
[iter 0] arena: candidate win-rate vs best = 1.00
[iter 0] candidate PROMOTED to best
  saved checkpoints/best.pt
```

---

## Training

```bash
uv run python -m scripts.train_main --config configs/default.yaml
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir runs
# open http://localhost:6006
```

Logged scalars: `train/loss`, `train/policy_loss`, `train/value_loss`,
`arena/win_rate_vs_best`, `buffer/size`, `selfplay/positions`.

Checkpoints:
- `checkpoints/best.pt` — the current best network (overwritten on each promotion)
- `checkpoints/best_iterN.pt` — snapshot every 5 iterations

Resume from a checkpoint:

```bash
uv run python -m scripts.train_main --config configs/default.yaml --resume checkpoints/best.pt
```

---

## Playing against the agent

```bash
uv run python -m scripts.play_human --checkpoint checkpoints/best.pt --human-color black
```

- Moves are entered **GTP-style**: `A1`, `C3`, `B5`, or `pass`.
- The `I` column is skipped, per Go convention (columns are `A B C D E F G H J K...`).
- `--sims` overrides MCTS search strength (more sims = stronger, slower).
- The agent uses MCTS with temperature 0 (greedy) and no Dirichlet noise.

Example session:

```
   A B C D E
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .
   A B C D E
  to move: Black (X)

your move > C3
agent played C5
   A B C D E
 5 . O o . .
 4 . . . . .
 3 . . X . .
 ...
```

Lowercase letters mark the most recent move.

---

## Watching games

```bash
# self-play game (same net for both colors)
uv run python -m scripts.watch_game --black checkpoints/best.pt --white checkpoints/best.pt

# pit two different checkpoints
uv run python -m scripts.watch_game --black checkpoints/best_iter0.pt --white checkpoints/best.pt

# dump a PNG per move
uv run python -m scripts.watch_game --save-pngs frames/
```

**Handicapping a side without a second network.** `--black-sims` and
`--white-sims` give each player a different MCTS simulation budget from the
*same* checkpoint — more simulations means deeper search and stronger play. This
lets you pit "dumber Black" against "smarter White" to probe how search depth
trades off against Go's first-move advantage:

```bash
# White thinks 2× harder than Black, same weights
uv run python -m scripts.watch_game --black-sims 32 --white-sims 64 | tail -1
```

---

## The handicap study: how big is the first-move advantage?

On a 5×5 board, **Black (who moves first) has a large, structural advantage** —
small boards make the opening move extremely valuable, and komi only partly
compensates. With equal search, self-play games are *not* a coin flip: this
agent wins as Black ~**97%** of the time. (This is exactly why the training
[arena](#the-arena-promotion-gate) swaps colors — to cancel the bias when
comparing two networks.)

That raises a fun question: **how badly do we have to handicap Black's search
before White's deeper thinking actually overcomes the first-move edge?** Because
both players can run from the *same checkpoint* at different simulation counts,
we can measure it directly. `scripts/handicap_study.py` fixes White's budget,
sweeps Black's over a fine grid, plays many randomized-opening games at each
point, and plots Black's win-rate and mean score margin:

```bash
uv run python -m scripts.handicap_study --white-sims 64 --games 16
# -> study/handicap_white64.png  +  study/handicap_white64.csv
```

### Result: White = 64 simulations/move

![Handicap study, White fixed at 64 sims](study/handicap_white64.png)

The crossover is **startlingly low**. Against White's 64 simulations, Black
reaches a break-even win-rate at only **~5 simulations** — a **~13× search
deficit** is needed just to bring the game back to even. Black only *loses the
majority* once its search collapses to **≤4 sims** (essentially playing its raw
policy with almost no lookahead, win-rate `0–31%`). Give Black even a handful of
simulations and the first-move advantage reasserts itself; by ~24+ sims Black is
back to winning 80–100%.

| Black sims vs White's 64 | Black win-rate | mean score (Black) |
|---|---|---|
| 1  | 0%   | −8.6 |
| 4  | 25%  | −8.9 |
| **~5** | **~50% (crossover)** | **~0** |
| 8  | 75%  | +4.8 |
| 16 | 75%  | +1.3 |
| 64 (equal) | 100% | +7.8 |

The score margins carry large error bars (±5–14 points) — a single deterministic
game per setting would be far too noisy to see the trend, which is why each
point averages 16 randomized-opening games. The takeaway: **on a small board,
first-move advantage dominates a surprisingly wide range of search asymmetry.**

> **Coming soon:** the same sweep with **White = 128** simulations, to show how
> the crossover point *shifts* as the stronger side searches deeper. Reproduce
> any level yourself with `--white-sims <N>`.

---

## Configuration reference

Everything tunable lives in `configs/default.yaml`:

| section | key | meaning |
|---------|-----|---------|
| `game` | `board_size` | NxN board. Start small (5), scale up later. |
| | `komi` | White's compensation; half-integer avoids draws. |
| | `max_moves` | hard cap on moves per game (prevents infinite loops). |
| `model` | `in_channels` | input planes (must match `encoding.NUM_PLANES`). |
| | `num_res_blocks` | depth of the conv trunk. |
| | `channels` | conv filter width. |
| | `value_channels` | filters in the value head's 1×1 conv. |
| | `hidden_size` | value head's hidden FC size. |
| `mcts` | `num_simulations` | MCTS rollouts per move. Higher = stronger, slower. |
| | `c_puct` | PUCT exploration constant. |
| | `dirichlet_alpha` | root prior noise concentration (Go: ~0.03·361/N² heuristic). |
| | `dirichlet_frac` | weight of Dirichlet noise at the root. |
| | `temp_moves` | use temperature=1 for the first N moves, then greedy. |
| `selfplay` | `num_parallel_games` | concurrent games batched through the GPU. |
| | `target_games` | games generated per self-play round. |
| `train` | `buffer_size` | max positions retained (FIFO when full). |
| | `batch_size` | minibatch per gradient step. |
| | `num_steps` | gradient steps per training round. |
| | `lr` | Adam learning rate. |
| | `l2_weight` | L2 regularization (also Adam weight_decay). |
| | `policy_weight` / `value_weight` | loss component weights. |
| `arena` | `num_games` | games to evaluate candidate vs best. |
| | `win_threshold` | candidate must beat best by this fraction to be promoted. |
| `loop` | `num_iterations` | self-play → train → arena rounds. |
| | `eval_every` | run arena every N iterations. |
| `runtime` | `device` | `cuda` or `cpu`. |
| | `seed` | RNG seed for reproducibility. |

---

## Scaling to bigger boards

To move from 5×5 to 7×7 or 9×9, **edit `configs/default.yaml` only**:

```yaml
game:
  board_size: 9        # was 5
  komi: 6.5            # standard 9×9 komi
  max_moves: 200       # was 60
model:
  channels: 128        # was 64 — bigger board needs more capacity
  num_res_blocks: 6    # was 4
mcts:
  num_simulations: 128 # was 64 — more search for the bigger tree
selfplay:
  num_parallel_games: 64  # was 128 — lower if VRAM gets tight
```

No code changes are needed — board size, action size, and network shapes all
derive from the config. When VRAM is tight, reduce `num_parallel_games` or
`batch_size` first; those are the main memory consumers.

---

## Performance & GPU notes

Measured on an RTX 5060 Ti (16 GB), 5×5, 128 parallel games, 64 simulations:

| metric | value |
|--------|-------|
| network params | ~310k |
| self-play throughput | ~126 positions/s (4.2 games/s) |
| peak GPU memory | ~20 MB |

**Important:** at 5×5 the network is so small that **MCTS is CPU-bound**, not
GPU-bound — the GPU is mostly idle. The bottleneck is the Python tree
traversal and `board.play` rule checking. The GPU becomes the real workhorse as
the board and network grow (the same batching gives much bigger relative wins
at 9×9+).

Highest-leverage speedups if you want more throughput at larger boards:
1. **Vectorize the board/group logic** with NumPy (flood-fill on arrays instead
   of Python sets) — biggest CPU win.
2. **Add virtual loss** so multiple leaves per game can be evaluated in the
   same GPU call.
3. **Increase `num_parallel_games`** to amortize each forward pass over more
   leaves (the 16 GB card has plenty of headroom).

---

## Limitations & future work

This is a pedagogical, from-scratch implementation. Known simplifications:

- **Simple ko only** (no positional superko). Sufficient for self-play learning;
  not tournament-legal in all cases.
- **Area scoring** with komi (no dead-stone removal at scoring time). On small
  boards with self-play this is rarely an issue, but real Go needs a
  dead-stone handling pass.
- **No resignation** (reserved in config but not wired up) and no temperature
  annealing beyond the hard `temp_moves` cutoff.
- **CPU-bound MCTS at small board sizes** — see performance notes above.

Natural next steps:
- Scale to 9×9 with a bigger net and more simulations.
- Add virtual loss + a vectorized board for 5–10× throughput.
- Add resign/temperature annealing and a learning-rate schedule.
- Add positional superko for full rule compliance.
- Track ELO over iterations via a ladder of checkpoints.

---

## License

MIT — see `LICENSE` if present, otherwise treat as MIT for personal/educational
use. This is an independent reimplementation for learning; it is not affiliated
with or endorsed by DeepMind.
