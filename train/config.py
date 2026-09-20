"""Single source of truth for network shapes and the action space.

Both the JAX trainer (`train/net.py`) and the torch deployment twin
(`train/net_torch.py`) build their parameters from `param_shapes(cfg)`, so the
two cannot drift: a flat dict keyed by '/'-joined paths, stored in JAX
convention (conv HWIO, linear (in, out)).  The twin transposes 4-D weights on
load.

Trunk: a 3x3-patch ViT (49 tokens) after Straka et al. (arXiv:2606.23348),
with RMSNorm pre-norm, QK-norm and SwiGLU.

Why not 1x1 patches: 441 tokens costs ~17 GFLOP and fails the 150 ms CPU move
budget, and buys nothing -- a 3x3 patch of 33 planes is 297 values projected
into 448 dims, an over-complete embedding, so no cell-level information is
lost at the input.
"""
from __future__ import annotations

from dataclasses import dataclass

B = 21                      # padded board side
N_CELLS = B * B             # 441
N_DIRS = 4                  # up, down, left, right
PATCH = 3
GRID = B // PATCH           # 7
N_TOKENS = GRID * GRID      # 49, the patch grid -- spatial, used by
                            # patchify/unshuffle/pos, not the sequence length

# --- temporal encoder, after Straka et al.'s TemporalEncoder ---------------
# Five aggregate series over time.  All five are exactly observable: the engine
# hands us `opponent_land_count` and `opponent_army_count` as scalars, the same
# scoreboard a human sees.  Enemy castles are deliberately absent: a structure
# in fog is indistinguishable from a mountain, so an enemy castle count cannot
# be known, and five exact series carry no fog-leak surface at all.
SERIES = ("own_army", "own_land", "own_castles", "enemy_army", "enemy_land")
N_SERIES = len(SERIES)      # 5

# Spatial history is a bank of EMAs rather than stacked frames.  Effective
# window ~1/alpha: 2, 8, 32, 128 ticks.  With an EMA the horizon comes from the
# decay rate, not the plane count, so four planes per field reach 128 ticks
# where seven stacked frames reach seven.
EMA_ALPHAS = (0.5, 0.125, 0.03125, 0.0078125)
N_EMA = len(EMA_ALPHAS)                 # per field
EMA_FIELDS = ("own_army", "ghost_army")

# --- backward curriculum depth ---------------------------------------------
# Ticks before the tagged scenario moment at which a seeded episode may start.
# Depth 0 is the tagged moment itself, where the opportunity already exists, so
# it can only teach conversion.  A deeper start hands the policy the same
# position minus the advantage: it has to manufacture the opportunity before
# converting it.
#
# The episode end is held fixed at (tagged tick + scenario horizon) regardless
# of depth, so a deeper start is strictly a longer task on the same target.
# That is what makes depths comparable.
SEED_DEPTHS = (0, 40, 80, 160)
N_DEPTH = len(SEED_DEPTHS)

# Aggregate series live in multi-scale rings, not an exponential resample.
# Four rings of 16, pushed when t % stride == 0, cover 16 / 64 / 256 / 1024
# ticks at 1 / 4 / 16 / 64 tick resolution.  The seed bank forces this shape: a
# seeded episode starts mid-game and must continue the series, so the
# representation has to be O(1) updatable and storable.  A strided resample of
# a raw history buffer is neither -- it needs the whole preceding buffer at
# injection.
TSCALES = (1, 4, 16, 64)
TRING = 16                  # samples per scale
N_TSAMP = TRING * len(TSCALES)          # 64 samples per series
N_TEMPORAL = 2              # tokens appended, as Straka et al.
N_SEQ = N_TOKENS + N_TEMPORAL           # 51 -- the transformer sequence length

# Rollout geometry, Straka et al. Table V.  The trainer and every profiling
# tool read these, so a tool can never measure a geometry the trainer does not
# use.
ENVS = 512          # environments per iteration
SEG = 512           # rollout ticks per iteration -> 262,144 transitions
MINIBATCH = 1024    # minibatch size in transitions (not a minibatch count)
EPOCHS = 1          # one pass over the filtered batch

# competition rules
DEATHTOUCH_TURN = 800       # from here, any move onto the enemy general wins
# Feature normalisation.  Defined here because obs.py, np_obs.py and the seed
# builder all consume it; a drifted scale would make injected seeds subtly
# wrong with no shape error to catch it.
ARMY_SCALE = 6.0            # log1p(200) ~ 5.3
TOTAL_SCALE = 8.0
AGE_SCALE = 48.0
CASTLE_SCALE = 12.0         # castles are built in this ruleset; 12 is a full board

BONUS_PERIOD = 50           # engine: owned cells grow when time % 50 == 0
STRUCT_PERIOD = 2           # engine: structures grow when time % 2 == 0
DRAW_TURN = 1200            # truncation: the game is a draw

# --- action space: cell x intent, plus the build this ruleset adds ----------
#
#   one categorical over (source cell, intent), N_SRC * N_INT = 4410 logits
#
# The acting cell is named explicitly rather than through abstractions such as
# "largest stack" slots or BFS-routed target cells.  An abstract source or
# target resolves to a concrete move from board state at execution time, so its
# legality mask can only be marginal -- "some source can act", not "this action
# is legal" -- and a legal-looking choice can silently become a pass.  Worse,
# that failure is unlearnable: the policy neither picks nor predicts the
# resolved cell, so no gradient path to avoiding the waste exists.  With an
# explicit cell the mask is exact, and whatever the argmax picks the engine
# executes.  The pathfinding knowledge the abstractions carried lives in the
# geodesic observation planes instead, where no mask problem can arise.
N_SRC = N_CELLS                     # 441: the acting cell, named explicitly

# Intents mirror Straka et al.'s 9 (4 directions x {all, half}, plus one) and
# add the build their ruleset does not have.  The engine takes the half-split
# natively (`split_army == 1` moves half the stack).
INT_DIR0 = 0                        # 0..3: direction, move all but one
INT_HALF0 = N_DIRS                  # 4..7: same direction, move half
INT_BUILD = 2 * N_DIRS              # 8: build at the named cell
INT_PASS = INT_BUILD + 1            # 9: end the turn
N_INT = INT_PASS + 1                # 10

N_JOINT = N_SRC * N_INT             # 4410: one categorical

# --- observation planes.  Order is load-bearing: obs.py (jax), the numpy
# deployment twin and every test index these by name via PLANES.
PLANES = (
    # visible now
    "own_cells", "enemy_cells", "neutral", "mountains",
    "own_castles", "other_castles", "own_general", "fog",
    "structures_in_fog", "ever_seen",
    "own_army", "enemy_army",
    # memory substrate -- under fog the current frame is not Markov
    "ghost_owner", "ghost_army", "staleness",
    # sticky: generals never move, and strong bots convert a sighting into a
    # kill within a few ticks -- forgetting one would be fatal
    "enemy_general_seen",
    # Multi-timescale motion.  Stacking consecutive frames gives a window of a
    # few ticks -- under a quarter of a board traverse (20-40 ticks) -- and
    # adjacent frames are ~99.5% identical because one move changes ~2 cells of
    # 441.  Exponentially strided snapshots would fix the coverage, but each
    # plane would refresh only every 2^k ticks, so its age slides 0..2^k
    # unannounced: phase-locked aliasing in a game that already has 50- and
    # 2-tick periodicities.  EMAs avoid both: they update every tick, cost the
    # same O(1) state, and `x - E_k` is a band-passed motion signal.  The trade
    # is smoothed history instead of exact snapshots.
    "ema_own_2", "ema_own_8", "ema_own_32", "ema_own_128",
    "ema_enemy_2", "ema_enemy_8", "ema_enemy_32", "ema_enemy_128",
    # Geodesic fields: step distance over terrain the agent has actually seen
    # (`obs.mountains` is fog-masked by the engine, so this leaks nothing).  A
    # 7-layer trunk gives ~7 rounds of propagation while real paths reach 55
    # steps, so the network cannot represent a geodesic and would have to
    # approximate it -- worst exactly around mountains, where paths are
    # contested.  The three fields serve the kill race, garrison timing and
    # the expansion frontier.
    "dist_enemy_general", "dist_own_general", "dist_frontier",

    # action cost (same category as a legal-move mask, not a feature)
    "build_price",
    # No game-clock plane.  A ticks-remaining input becomes a crutch: the
    # policy leans on it instead of learning urgency, and it is an
    # out-of-distribution hazard whenever the training cap differs from the
    # deployed game length.  With the draw priced at -1, urgency has to be
    # internalised.
    #
    # Economy clocks stay.  The engine grows every owned cell when
    # time % 50 == 0 and every structure when time % 2 == 0.  Late game the
    # 50-tick bonus is a 60-100 army step on a known tick, so whether to trade
    # now or after it is a real decision -- and without a phase feature the
    # network cannot represent "three turns before the bonus".  Encoded as
    # time remaining, not elapsed: with gamma=1 what prices a state is
    # time-to-event.  Small value = imminent.
    "bonus_in", "struct_in",
    "own_land_n", "own_army_n", "enemy_army_n",
)
N_PLANES = len(PLANES)      # 33

# --- storage groups ---------------------------------------------------------
# The trajectory buffer is (seg, 2*envs, N_PLANES, B, B) fp16 and it dominates
# device memory -- it, not FLOPs, is what caps `envs`.  Stored naively it wastes
# more than half its bytes on representation: twelve planes are boolean but
# hold 16 bits each, and five are broadcast scalars stored as 441 identical
# copies.  Both are recoverable losslessly, so the fix is the encoding, not the
# feature set.
_BOOL = ("own_cells", "enemy_cells", "neutral", "mountains", "own_castles",
         "other_castles", "own_general", "fog", "structures_in_fog",
         "ever_seen", "ghost_owner", "enemy_general_seen")
_SCALAR = ("bonus_in", "struct_in", "own_land_n", "own_army_n",
           "enemy_army_n")
BOOL_IDX = tuple(PLANES.index(n) for n in _BOOL)
SCALAR_IDX = tuple(PLANES.index(n) for n in _SCALAR)
FLOAT_IDX = tuple(i for i in range(N_PLANES)
                  if i not in BOOL_IDX and i not in SCALAR_IDX)
N_BOOL, N_SCALAR, N_FLOAT = len(BOOL_IDX), len(SCALAR_IDX), len(FLOAT_IDX)
N_PACKED = (N_BOOL + 7) // 8            # uint8 planes after bitpacking

# privileged planes, critic-only, never reach the actor's input
PRIV_PLANES = ("enemy_general", "true_enemy_army", "true_enemy_owner")
N_PRIV = len(PRIV_PLANES)


@dataclass(frozen=True)
class NetCfg:
    """3x3-patch ViT.  Defaults are Straka et al.'s depth and width."""
    # Straka et al. Table II: depth 7, embedding 448, feedforward 1344,
    # 8 heads.  Their feedforward is a 2-matrix GELU MLP at 3x, so 2*448*1344 = 1.204M
    # MAC/token.  SwiGLU uses three matrices, so the equal-FLOP width is 2.0x
    # (H=896): 3*448*896 = 1.204M, identical.
    dim: int = 448           # D
    depth: int = 7           # L
    heads: int = 8           # head_dim = dim // heads = 56
    mlp_mult: float = 2.0    # SwiGLU; the 3-matrix equivalent of GELU at 3x
    cell_dim: int = 128      # Dc, per-cell features after the pixel shuffle
    dec_convs: int = 1       # full-resolution 3x3 refinements after the shuffle
    temp_hidden: int = 256   # TemporalEncoder width
    priv_hidden: int = 128   # privileged plane encoder width (critic only)
    q_dim: int = 256         # Q-critic width; training-only, so cost is cheap
    q_rank: int = 8          # rank of the source x intent interaction in Q
    bf16: bool = True        # bf16 matmuls, f32 master weights and norms

    @property
    def head_dim(self) -> int:
        assert self.dim % self.heads == 0
        return self.dim // self.heads

    @property
    def mlp_hidden(self) -> int:
        """Rounded to a multiple of 64 -- matmul shapes the GPU likes."""
        return int(round(self.dim * self.mlp_mult / 64)) * 64


def _lin(s, name, cin, cout, bias=True):
    s[name + "/w"] = (cin, cout)
    if bias:
        s[name + "/b"] = (cout,)


def _rms(s, name, d):
    s[name + "/g"] = (d,)


def param_shapes(cfg: NetCfg) -> dict[str, tuple[int, ...]]:
    """Flat '/'-joined path -> shape, in JAX convention (conv HWIO)."""
    D, H, Dc, hd = cfg.dim, cfg.mlp_hidden, cfg.cell_dim, cfg.head_dim
    s: dict[str, tuple[int, ...]] = {}

    # --- trunk
    _lin(s, "patch", PATCH * PATCH * N_PLANES, D)
    s["pos"] = (N_TOKENS, D)
    # TemporalEncoder: the five aggregate series -> N_TEMPORAL tokens appended
    # to the patch sequence.  `ttype` distinguishes the tokens from each other;
    # without it the two are symmetric and the encoder can only emit the same
    # vector twice.
    _lin(s, "temp/in", N_SERIES * N_TSAMP, cfg.temp_hidden)
    _lin(s, "temp/out", cfg.temp_hidden, N_TEMPORAL * D)
    s["ttype"] = (N_TEMPORAL, D)
    for i in range(cfg.depth):
        p = f"blk{i}/"
        # LLaMA-style: biasless inside the block.  QK-norm is an RMSNorm over
        # the head dimension, shared across heads -- the cheapest guard against
        # attention-logit blowup.
        _rms(s, p + "n1", D)
        _lin(s, p + "qkv", D, 3 * D, bias=False)
        _rms(s, p + "qn", hd)
        _rms(s, p + "kn", hd)
        _lin(s, p + "proj", D, D, bias=False)
        _rms(s, p + "n2", D)
        _lin(s, p + "gate", D, H, bias=False)
        _lin(s, p + "up", D, H, bias=False)
        _lin(s, p + "down", H, D, bias=False)
    _rms(s, "ln_f", D)

    # --- decoder: 49 tokens -> 441 cells (pixel shuffle), then full-resolution
    # 3x3 convs so a cell logit has a real per-cell receptive field rather than
    # only its patch's token.
    _lin(s, "dec", D, PATCH * PATCH * Dc)
    for j in range(cfg.dec_convs):
        s[f"dcv{j}/w"] = (3, 3, Dc, Dc)
        s[f"dcv{j}/b"] = (Dc,)
    _rms(s, "dec_n", Dc)

    # --- policy heads
    _lin(s, "cell_int", Dc, N_INT)  # N_INT logits per cell

    # --- training-only heads.  The deployment twin ignores these.
    _lin(s, "gen", Dc, 1)           # aux: enemy-general location
    _lin(s, "danger", D, 1)         # aux: death within 32 ticks

    # --- Q-critic, centralized.  Q must be evaluated at every action to form
    # Vbar = sum_a pi(a|s) Q(s,a), so a joint 441x10 table is out.  We use an
    # additive decomposition plus a rank-r bilinear interaction:
    #
    #   Q(s, i, j) = q0 + qs[i] + qi[j] + <u[i], v[j]>
    #
    # which keeps Vbar exact and cheap for a factored policy:
    #
    #   Vbar = q0 + pi_s.qs + pi_i.qi + <pi_s U, pi_i V>      O(r(441+10))
    #
    # r=0 is the pure additive (dueling) form; r>0 lets Q say "north from cell
    # 137 is good, north from cell 200 is not", which additive-only cannot
    # express.
    r = cfg.q_rank
    s["priv/c/w"] = (3, 3, N_PRIV, cfg.priv_hidden)
    s["priv/c/b"] = (cfg.priv_hidden,)
    _lin(s, "q/mix", Dc + cfg.priv_hidden, cfg.q_dim)
    _lin(s, "q/src_cell", cfg.q_dim, 1 + r)
    _lin(s, "q/intg", cfg.q_dim, N_INT * (1 + r))
    _lin(s, "q/base", cfg.q_dim, 1)
    return s


TRAIN_ONLY = ("gen", "danger", "priv", "q")


def n_params(cfg: NetCfg, deploy_only: bool = False) -> int:
    """Total parameters; `deploy_only` counts just the policy path that ships."""
    out = 0
    for k, shp in param_shapes(cfg).items():
        if deploy_only and k.split("/")[0] in TRAIN_ONLY:
            continue
        n = 1
        for x in shp:
            n *= x
        out += n
    return out


def gflops(cfg: NetCfg) -> float:
    """Analytic forward FLOPs (MAC = 2) at batch 1, policy path only.

    N_TOKENS vs N_SEQ matters here: the patch embed and the pixel shuffle are
    per patch (49), while the transformer blocks run over the full sequence
    including the temporal tokens (51).  Charging the blocks at 49 would
    under-report the trunk by 4%.
    """
    D, H, Dc = cfg.dim, cfg.mlp_hidden, cfg.cell_dim
    N, S = N_TOKENS, N_SEQ
    f = 2 * N * (PATCH * PATCH * N_PLANES) * D                  # patch embed
    f += 2 * (N_SERIES * N_TSAMP) * cfg.temp_hidden             # TemporalEncoder
    f += 2 * cfg.temp_hidden * (N_TEMPORAL * D)
    per_layer = 3 * D * D + D * D + 2 * S * D + 3 * D * H       # qkv, proj, attn, swiglu
    f += 2 * S * cfg.depth * per_layer
    f += 2 * N * D * (PATCH * PATCH * Dc)                       # pixel shuffle
    f += cfg.dec_convs * 2 * N_CELLS * 9 * Dc * Dc              # refinement convs
    f += 2 * N_CELLS * Dc * N_INT                               # policy head
    return f / 1e9
