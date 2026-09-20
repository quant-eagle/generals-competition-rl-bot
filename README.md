# generals-zero

Self-play reinforcement learning for [generals.io](https://generals.io) in JAX.
A 10M-parameter vision transformer is trained from a sparse win/loss signal,
with the simulator, both players' forward passes, the advantage computation and
the optimizer compiled into a single on-device step.

**Result.** Trained from scratch for roughly 24 hours on 8× RTX 5090, the agent
reached #10 on the [generals.bot](https://www.generals.bot) competition ladder.
It plays greedy argmax on one CPU core inside the competition's 150 ms move
budget and 50 MB submission limit.

The full design rationale is in [docs/DESIGN.md](docs/DESIGN.md).

## The problem

Generals.io is a two-player, simultaneous-move strategy game on a grid: expand,
grow armies, find the enemy general under fog of war, and capture it.

- **Imperfect information.** The current frame is not Markov; what was seen
  fifty ticks ago, and where the enemy's army probably is, decide games.
- **A large action space.** Every tick: which of 441 cells acts, and how.
- **Long horizons, sparse reward.** Games run for hundreds of ticks and the only
  ground-truth signal is who captured whom.
- **A deployment budget.** One CPU core, 150 ms per move, a 50 MB zip. This
  bounds the model, and so shapes everything upstream of it.

## System

```
                        one compiled train step (jit / shard_map)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  board pool ─┐                                                           │
 │  seed bank ──┼─► lax.scan over 512 ticks × 512 envs × 2 seats            │
 │              │     engine step ─► observation + fog memory ─► ViT policy │
 │              │     privileged Q-critic ─► Q(s,a), V̄(s) stored            │
 │              │                                                           │
 │              └─► Q-boosted advantages ─► keep top 25 % by |A|            │
 │                  ─► PPO update (1 epoch) ─► EMA of weights               │
 └──────────────────────────────────────────────────────────────────────────┘
        host: curriculum gates · evaluation · checkpoints · logging
```

No transition data crosses to the host. It schedules curricula, runs evaluation
and writes checkpoints.

## Prior work and what this repository adds

The backbone of the method is published work:

- **From Straka et al. [1]:** the 3×3-patch ViT trunk and its temporal tokens,
  the cell × direction action grid, sparse terminal reward, top-advantage
  filtering, the parameter EMA, the entropy and learning-rate schedules, and the
  skeleton of the scripted expander prior. The simulator is theirs as well.
- **From Fan & Farina [2]:** the Q-boosted advantage estimator (VRPO) that
  replaces GAE.
- **From Salimans & Chen [4]:** the idea of a backward curriculum from
  demonstration states.

What is specific to this repository:

- **Multi-timescale EMA planes in place of stacked frames.** Straka et al.
  give the network motion by stacking consecutive frames, which covers a window
  as long as the stack and spends most of its planes on near-identical images.
  Here the spatial history is a bank of exponential moving averages of army
  strength at α = 1/2, 1/8, 1/32 and 1/128, so the horizon is set by the decay
  rate rather than the plane count: four planes per field reach ~128 ticks.
  The EMAs live in the same units as the current-army planes, so `x − E_k` is a
  band-passed motion signal the trunk can form with one subtraction, at four
  speeds at once. They update every tick, which avoids the aliasing a strided
  snapshot would have against the game's 2-tick and 50-tick economy cycles, and
  they are O(1) state, so a seeded mid-game episode can carry them. Applied to
  the *remembered* enemy army, the same filter yields a different quantity:
  the ghost value is frozen while a cell is in fog, so `ghost − E_k(ghost)`
  measures how much a fresh sighting revises what was believed for the last k
  ticks.
- **A factored Q-critic that keeps the policy expectation exact.**
  `Q(s,i,j) = q0 + qs[i] + qi[j] + ⟨u[i], v[j]⟩` at rank 8 makes
  V̄(s) = Σ π·Q computable in O(r·(441+10)) instead of over 4,410 actions, which
  is what makes Q-boosting affordable on this action space. The critic is
  centralized: it sees privileged enemy planes that never reach the actor.
- **Geodesic observation planes.** BFS step distance over known terrain to the
  remembered enemy general, the own general and the expansion frontier, which a
  shallow trunk cannot compute for itself; and no game-clock plane.
- **A backward curriculum with exact belief state.** Seeded mid-game positions
  carry reconstructed fog memory, motion EMAs and scoreboard history for both
  seats, stored in O(1)-updatable multi-scale rings so a seeded episode continues
  its history rather than starting from a blank one. Start depths advance per
  scenario with the episode end held fixed.
- **An exact legality mask with a build intent**, so every action with
  probability mass is one the engine executes.
- **A profile-driven attention formulation.** Sub-layer ablation showed
  attention taking 48 % of the forward's time while owning about 36 % of its
  FLOPs, and that the gap was data movement rather than arithmetic: holding
  q/k/v as (tokens, heads, dim) makes XLA move the whole tensor around every
  einsum, and the backward repeats it. The block instead does one transpose to
  (3, heads, tokens, head_dim), so both attention products compile to plain
  batched matmuls over a leading head axis, and it computes the QK scores in
  bf16 with only the softmax in f32, since q and k already come out of a bf16
  matmul and upcasting them recovers no precision. Identical arithmetic;
  +4.0 % training throughput in a controlled A/B at fixed architecture and
  +6.5 % on the 448 / 7 trunk (single RTX 5090), most of it in the backward.
- **Throughput engineering.** A lossless packed trajectory buffer (bit-packed
  booleans, scalars stored as scalars) that more than halves the memory that
  caps environment count; one forward for both seats; `shard_map` data
  parallelism measured at ≈3.4× on four GPUs; a pipelined host metric drain.
- **A deployment path with tested parity.** A PyTorch twin of the network and
  NumPy twins of the observation, mask and executor, all built from one shape
  table, with logit and observation parity pinned by tests.

## Getting started

The simulator is [generals-bots](https://github.com/strakam/generals-bots) by
Matej Straka, installed separately. The code is tested against the pinned
commit:

```bash
git clone https://github.com/strakam/generals-bots engine
git -C engine checkout 9e3b9d1
python -m venv .venv && source .venv/bin/activate
pip install -e engine -e ".[dev]"      # install a CUDA build of jax for GPU training
make test
```

Train, evaluate, ship:

```bash
python -u -m train.train --no-seeds --dim 384 --depth 6 --devices 8
python -u tools/evaluate.py --ckpt train/checkpoints/ckpt_0020000.npz
python -u tools/h2h.py --help                          # checkpoint vs checkpoint
python -u tools/package_submission.py --ckpt train/checkpoints/ckpt_0020000.npz
python -u tools/latency_full.py --zip dist/submission.zip
```

`--no-seeds` runs pure self-play and needs no replay data; pass `--seeds` once a
seed bank is built. `--resume` continues the latest checkpoint. Set
`WANDB_API_KEY` in the environment or a git-ignored `.env` to log to Weights &
Biases; without it, training logs to stdout only.

## Limitations

- **The ladder result is not reproducible from this repository alone.** It used
  the backward curriculum, whose seed bank is built from ladder replays that are
  not distributed here. `--no-seeds` is tested to train end to end, but its
  playing strength has not been measured.
- **One run.** There are no repeated seeds and no published ablations, so the
  contribution of each component in [docs/DESIGN.md](docs/DESIGN.md) is argued
  from design, not demonstrated.
- **Ladder rank is a noisy, point-in-time measure** against a field that keeps
  changing.
- **No exploitability estimate.** Mirror self-play can cycle; the mitigations
  here are a scripted-opponent league and a fixed-opponent evaluation sidecar
  (`tools/fixed_eval.py`), which detect regressions against known strategies but
  do not bound exploitability. Deployment is deterministic (greedy), which is
  exploitable in principle.
- **The tables list code defaults, not a run log.** The ladder run did not use
  all of them. Draw pricing, the training clock cap, the league fraction and
  build-exploration epsilon are flags.

## Repository layout

```
train/            the package
  config.py         shapes, planes, action space: the single source of truth
  net.py            JAX ViT, policy head, privileged Q-critic, aux heads
  net_torch.py      PyTorch deployment twin
  obs.py / np_obs.py    observation, fog memory, legality mask (JAX / NumPy)
  exec.py / np_exec.py  action executor (JAX / NumPy)
  rollout.py        on-device self-play scan, league, seeded episodes
  learn.py          Q-boosted advantages, filtering, PPO loss, schedules
  magnet.py         scripted expander prior
  boards.py         board curriculum
  seeds.py          seed bank and exact memory reconstruction
  parallel.py       shard_map data parallelism
  train.py          the training loop and CLI
  arena.py          greedy evaluation and play-style profile
  ckpt.py / wb.py   atomic resumable checkpoints, optional W&B logging
  submission/       competition entrypoint
tools/            evaluation, head-to-head, latency, profiling, packaging
data/             replay fetching and decoding pipeline
tests/            parity, design-invariant and learner tests
docs/DESIGN.md    design rationale
```

## References

1. M. Straka, V. Lisý, M. Schmid. *Superhuman AI for Generals.io Using
   Self-Play Reinforcement Learning.* arXiv:2606.23348.
2. Z. Fan, G. Farina. *GAE Falls Short in Imperfect-Information Self-Play
   Reinforcement Learning.* arXiv:2605.19235.
3. M. Rudolph et al. *Reevaluating Policy Gradient Methods for
   Imperfect-Information Games.* arXiv:2502.08938.
4. T. Salimans, R. Chen. *Learning Montezuma's Revenge from a Single
   Demonstration.* arXiv:1812.03381.
5. S. Sokota et al. *A Unified Approach to Reinforcement Learning, Quantal
   Response Equilibria, and Two-Player Zero-Sum Games.* arXiv:2206.05825.

## License

MIT. See [LICENSE](LICENSE).
