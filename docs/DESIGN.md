# Design

How generals-zero is built and why. The [README](../README.md) has the summary,
the split between prior work and what this repository adds, and the limitations.
Reference numbers ([1]–[5]) are listed at the end.

## Network

A **3×3-patch vision transformer** over the 21×21 padded board, after Straka
et al. [1]:

| | |
|---|---|
| tokens | 49 patch tokens + 2 temporal tokens |
| width / depth / heads | 384 / 6 / 8 (head dim 48) |
| block | pre-norm RMSNorm, QK-norm, SwiGLU (hidden 768), bias-free |
| decoder | linear → pixel shuffle to 21×21×128 → 3×3 conv → RMSNorm |
| parameters | 9.95M total, 9.86M shipped (≈20 MB in fp16) |
| cost | 1.11 GFLOP per forward at batch 1 |

This is the model behind the ladder result (`--dim 384 --depth 6`). The code
default is the larger 448 / 7 configuration of Straka et al. (15.3M parameters,
1.66 GFLOP).

**Why 3×3 patches.** One token per cell would be 441 tokens and roughly
17 GFLOP, far over the CPU move budget. A 3×3 patch of 33 planes is 297 values
projected into 384 dimensions, so the patch embedding is over-complete rather
than a bottleneck. Per-cell resolution is restored at the output
by the pixel-shuffle decoder and a full-resolution convolution, so each cell's
logits have a genuine per-cell receptive field rather than sharing their
patch's token.

**Normalisation and precision.** QK-norm bounds the attention logits, a
precaution against logit growth under the non-stationary targets of self-play.
Matmuls run in bf16; master weights, every norm and the softmax stay
in f32.

**Temporal tokens.** Five scoreboard series (own army, land and castles; enemy
army and land) are kept in multi-scale ring buffers: 16 samples at each of 1,
4, 16 and 64-tick strides, covering 1,024 ticks in 64 numbers per series. A
small MLP turns them into two extra tokens. The ring representation is O(1) to
update and *storable*, which matters because seeded episodes (see
[Curricula](#curricula)) start mid-game and must carry their history with them.

**Training-only heads.** None of these ship:

- A **centralized Q-critic** that additionally sees three privileged planes
  (the true enemy general, armies and ownership). The privileged encoder feeds
  the critic only and never the actor.
- Two auxiliary predictions made *from the fogged trunk* and supervised by the
  hidden truth: where the enemy general is, and whether the agent dies within
  32 ticks. They push belief-state information into the features the policy
  actually uses.

## Observation

33 planes, grouped by what they are for:

- **Visible state (12)**: ownership, terrain, structures, armies, fog.
- **Fog memory (4)**: last-seen enemy ownership and army, a saturating
  staleness age, and a sticky enemy-general sighting. Generals never move, so
  one sighting should never be forgotten.
- **Motion (8)**: exponential moving averages of own and remembered-enemy army
  at α = 1/2, 1/8, 1/32, 1/128. Stacking consecutive frames is the usual
  answer, but adjacent frames are ~99.5 % identical and seven frames cover only
  seven ticks. With an EMA the horizon comes from the decay rate rather than the
  plane count: four planes reach ~128 ticks. EMAs also update every tick, which
  avoids the phase aliasing that strided snapshots would create against the
  game's own 2-tick and 50-tick economy cycles. The EMAs are kept in the
  same log-scaled units as the current-army planes, so `x − E_k` is a band-passed
  motion signal the trunk can form with a single subtraction. The two fields
  measure different things: own army is always visible, so its EMAs are true
  motion, while the remembered enemy army is frozen under fog, so
  `ghost − E_k(ghost)` is a belief-revision signal, the gap between a fresh
  sighting and what was believed for the last k ticks.
- **Geodesic fields (3)**: BFS step distance over *known* terrain to the
  remembered enemy general, the own general, and the nearest neutral cell. A
  6-layer trunk gets about six rounds of spatial propagation, while real paths
  around mountains reach 50+ steps, so the network cannot compute a geodesic
  internally, so the field is provided as input.
- **Costs and clocks (6)**: the castle build price per cell, time remaining to
  the next 50-tick land bonus and 2-tick structure growth, and three scoreboard
  scalars.

There is no game-clock plane. Training can run under a shorter time cap than
the competition's (`--train-cap`), and a policy that reads the clock would be
out of distribution once the cap changes; without the plane, time pressure has
to be inferred from the position.

All observation code exists twice: a JAX version inside the training scan and a
NumPy twin for deployment. Parity between them is pinned by tests.

## Action space

One categorical over **441 cells × 10 intents = 4,410 actions**: four
directions moving the full stack, four moving half, *build*, and *pass*. (The
half-move intents are part of the head but masked out in the default recipe;
top ladder play almost never splits.)

This is the cell × direction grid of Straka et al. with a *build* intent added
for the competition ruleset. A factored policy (pick a source, then pick an
intent) has fewer logits, but its legality masks can only be *marginal*: a
legal source paired with a legal intent can still be jointly illegal, and the
executor then silently passes. Those wasted turns are unlearnable, because the
policy never chose the thing that failed. A single joint head makes the mask
**exact**: whatever the argmax picks is an action the engine executes.

## Learning algorithm

PPO with two substitutions. Policy gradients rather than a
CFR- or search-based method is itself a choice: at this scale, generic policy
gradient methods match or beat the game-theoretic deep RL family on
imperfect-information benchmarks [3], and they parallelise trivially.

### Q-boosted advantages instead of GAE

Following Fan & Farina [2], the critic is a Q-function, and the advantage uses
the exact policy expectation as a per-step control variate:

```
V̄(s)  = Σ_a π(a|s) Q(s,a)
δ⁺_t  = r_t + γ V̄(s_{t+1}) − Q(s_t, a_t)
A_t   = Q(s_t,a_t) − V̄(s_t) + Σ_k (γλ)^k δ⁺_{t+k}
```

Each correction term has zero expectation under π, so the estimator stays
unbiased while the action-sampling noise of every future step is subtracted
out. A state-value critic cannot do this because it has nothing to take the
expectation over. Fan & Farina report that this is where GAE loses the most in
imperfect-information self-play.

Evaluating Q at all 4,410 actions per state would be prohibitive, so Q is
factored:

```
Q(s, i, j) = q0 + qs[i] + qi[j] + ⟨u[i], v[j]⟩        (rank 8)
```

which keeps V̄ exact in O(r · (441 + 10)), and means a single
sample supervises one source component and one intent component instead of one
entry of a 4,410-cell table.

### Top-advantage filtering

After computing advantages over the whole rollout, only the **top 25 % by |A|**
is trained on [1]. Most ticks in a strategy game are uneventful expansion;
filtering spends the update on the decisions with the largest estimated
effect. The update is a single epoch over the ~131k kept samples per device. Ranking by magnitude rather than
sign keeps the negative examples and avoids self-imitation.

### The rest of the recipe

Code defaults; each is a CLI flag.

| | |
|---|---|
| reward | sparse, terminal only: +1 / −1 (draw price configurable) |
| discount, trace | γ = 1, λ = 0.9 |
| PPO | clip 0.2, 1 epoch, minibatch 1,024, grad-clip 0.267 |
| entropy | 0.05 · (t+1)^−0.2, floored at 0.001 |
| learning rate | Adam, clip(0.5 · (t+1)^−1.1, 5e-6, 1e-4) |
| weights played | EMA (decay 0.99), which is what gets evaluated and shipped |
| sampling temperature | annealed 1.0 → 0.7 |

**Expander magnet.** Under a sparse terminal reward, self-play from random
weights produces almost no signal. A cross-entropy term pulls the policy toward a cheap
scripted prior ("push big stacks at neutral and enemy cells, build when it is
affordable, walk downhill toward a sighted general"), in the spirit of
magnetic mirror descent [5]. The prior's skeleton follows the training recipe
of Straka et al.; the hunt term, the build handling and the half-move discount
are additions for this ruleset. Its coefficient fades linearly to zero, so it
acts on early training only.

**Opponents.** Mirror self-play by default: both seats are the current policy,
computed in a single batched forward, and both seats' samples are trained on.
A `--league` fraction of games swaps in scripted opponents (their samples are
masked out of the loss), as a guard against forgetting simple strategies under
pure self-play. `--selfplay ladder` plays a FIFO of frozen snapshots
instead.

## Curricula

**Board curriculum.** Five stages from cramped 10–12-cell boards with generals
a few steps apart up to full competition maps. Small boards reach decisive
fights within tens of ticks, so the sparse reward is dense in practice while
the policy is weak. Promotion is gated on the EMA policy's win rate against a
scripted opponent, because a self-play win rate is 0.5 by construction and
cannot say whether the agent is competent. Boards are always padded to 21×21,
so a stage change never recompiles.

**Backward curriculum from expert positions.** A seed bank of mid-game states
from top-ladder replays, tagged by scenario (strike conversion, blind commit,
defence, castle decision), is mixed into the rollout at 30 %. Each seed carries
exact reconstructed fog memory, motion EMAs and scoreboard rings for both
seats, so the policy sees a state a real game could have produced. Following
Salimans & Chen [4], episodes first start *at* the decisive moment and then 40,
80 and 160 ticks before it, advancing per scenario as conversion improves, with
the episode end held fixed so depths stay comparable. Once every scenario is
mastered at full depth, the seed fraction anneals down.

Replay data and the seed bank built from it are not distributed with this
repository. `data/` contains the pipeline that fetches replays and decodes them
into engine-verified actions, and `tools/build_rl_seeds.py` attaches exact
memory to tagged positions. The trainer loads the bank from `--seeds`; without
one, `--no-seeds` trains on pure self-play with the board curriculum alone.

## Throughput

Wall-clock was the binding constraint. The measures taken, in rough order of effect:

- **Everything in one compiled step.** The JAX engine, observation and memory
  updates, both seats' forwards, the executor, the advantage scan, filtering
  and the optimizer run inside one `jit`. There is no host–device round trip
  per tick, and the host reads metrics one iteration late so the device never
  waits for it.
- **Attention written for the compiler.** Profiling at the sub-layer level
  showed attention taking 48 % of the forward's time for about 36 % of its
  FLOPs. The arithmetic was not the cause: removing the f32 score matmul saved
  2.2 % and removing QK-norm 1.7 %. The cause was layout. With q/k/v held as
  (tokens, heads, dim), XLA transposes a (batch, tokens, heads, head_dim) tensor
  around each einsum, four times per block, and again in the backward. One
  transpose to (3, heads, tokens, head_dim) up front turns both attention
  products into plain batched matmuls over a leading head axis. The QK scores
  are computed in bf16 and only the softmax runs in f32: q and k are outputs of
  a bf16 matmul, so upcasting them first buys no precision and forces a slower
  f32 matmul. The arithmetic is unchanged. Measured on one RTX 5090: +4.0 %
  training throughput in a controlled A/B at fixed architecture, +6.5 % on the
  448 / 7 trunk, with most of the gain in the update step because the backward
  carries more transposes than the forward. RMSNorm and QK-norm cost time for
  no FLOPs and were kept on purpose; they are the stability margin.
- **One forward for both seats.** Mirror self-play batches 2N observations
  through the trunk once per tick.
- **Trajectory storage, not FLOPs, is the memory limit.** Stored naively,
  twelve boolean planes cost 16 bits each and five broadcast scalars are stored
  441 times. Bit-packing the booleans and storing scalars as scalars cuts the
  buffer to under half, losslessly, which directly buys more environments per
  GPU.
- **A model sized for the deployment budget** is also a model that is cheap to
  roll out: the 49-token trunk is what makes 512 envs × 512 ticks × 2 seats per
  device per iteration affordable.
- **Data parallelism with `shard_map`.** Only the environment carry and RNG
  keys are sharded; parameters, optimizer state, board pool and seed bank are
  replicated, and gradients are `pmean`-ed before the optimizer so replicas
  stay bit-identical. The all-reduce is ~60 MB per step, which PCIe handles
  comfortably: measured on four RTX 5090s, throughput is ≈3.4× one card
  (86 % scaling efficiency); the ladder run used eight.
- **Sample efficiency.** Top-advantage filtering, the Q-boosted estimator, the
  magnet and both curricula are aimed at the number of transitions needed
  rather than the cost of one. Their individual contributions are not ablated
  here.

## Deployment

The competition bot is a stdin/stdout process on one CPU core. The shipped
path is a pure-functional **PyTorch twin** of the JAX network plus NumPy twins
of the observation encoder, legality mask and executor. Both networks build
their parameters from a single shape table (`train/config.py`), and tests pin
logit parity and observation parity between the trained and the shipped path.

`tools/package_submission.py` exports the EMA policy weights in fp16 (after
checking that fp16 does not change the greedy action), drops every
training-only head, vendors the runtime modules and smoke-tests the staged bot.
At run time a cheap heuristic move is computed first and the model overwrites
it only if it answers inside a 110 ms soft deadline, so a slow or failed
forward still yields a legal move.

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
