"""The magnet: a scripted expander prior the policy is regularised toward.

learn.py penalises KL(pi || m) = -H(pi) - sum pi log m.  The -H half is the
usual entropy bonus; the other half anchors the policy to the prior m defined
here, in the spirit of magnetic mirror descent (Sokota et al.,
arXiv:2206.05825) but with a fixed hand-written magnet instead of the previous
iterate.  The prior's skeleton follows the training recipe of Straka et al.
(arXiv:2606.23348); the three ruleset-specific terms below are additions.
Under a sparse terminal reward a newborn policy gets almost no signal, and the
prior hands it the one behaviour every opening needs -- expand, prefer valuable
cells, move the big stacks.  Its coefficient is annealed to zero (train.py), so
it is a scaffold, not a constraint.

    m(cell, intent) ~ src_score[cell] * int_score[cell, intent]

A move is scored by what stands on its destination: pass 0.2, default 1.0,
neutral 2.0, enemy 3.0, city S_CITY; the top-5 own stacks get +2.0 as sources.
Three terms are specific to this ruleset:

  * S_CITY rides on INT_BUILD.  Cities here are built, not captured, so
    building is the action that acquires one.  The score is a balance: high
    enough that greedy play keeps building at all, low enough that the prior
    does not subsidise castle farming over killing.  BUILD_CAP stops paying
    beyond the few castles strong players actually build.
  * S_HUNT multiplies directional intents whose destination decreases the
    `dist_enemy_general` plane.  Before first contact that plane is flat, so
    the term is exactly zero and the opening is pure expansion; once the
    general is localised the pull flips from "spread outward" to "converge".
    Occupancy alone cannot express this, because enemy cells are fog-hidden
    for most of the game.
  * S_HALF discounts half-moves, which top players use on 0-2% of turns;
    scored like full moves they soak up entropy and dilute attack chains.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import B, INT_BUILD, N_CELLS, N_DIRS, N_INT, N_PLANES, PLANES

NEG_INF = -1e30
_P = {name: i for i, name in enumerate(PLANES)}

DIRS_RC = ((-1, 0), (1, 0), (0, -1), (0, 1))
S_DEFAULT, S_NEUTRAL, S_ENEMY, S_CITY, S_PASS, S_TOPK = 1.0, 2.0, 3.0, 3.5, 0.2, 2.0
S_HUNT = 4.0
# Soft build cap: once the player owns BUILD_CAP castles the magnet stops
# paying for more (score falls to S_PASS).  A prior, not a rule -- a policy
# that truly profits from another castle can still learn it.
BUILD_CAP = 3
S_HALF = 0.3
TOP_K = 5


def expander_prior(planes, legal):
    """(N_PLANES, B, B), (N_CELLS, N_INT) -> log m over the joint grid.

        m(cell, intent) ~ src_score[cell] * int_score[cell, intent]

    Scoring the pair is what the flat policy allows: a (cell, direction) move
    is scored by what stands on the destination cell.
    """
    neutral = planes[_P["neutral"]]
    enemy = planes[_P["enemy_cells"]]
    castle = planes[_P["other_castles"]]
    own_army = planes[_P["own_army"]]          # log1p(army)/ARMY_SCALE

    occ = jnp.full((B, B), S_DEFAULT)
    occ = jnp.where(neutral > 0.5, S_NEUTRAL, occ)
    occ = jnp.where(enemy > 0.5, S_ENEMY, occ)
    occ = jnp.where(castle > 0.5, S_CITY, occ)

    dir_scores = jnp.stack([jnp.roll(occ, (-dr, -dc), (0, 1))
                            for dr, dc in DIRS_RC], -1).reshape(N_CELLS, N_DIRS)
    dir_scores = jnp.maximum(dir_scores, S_PASS)

    # Hunt term: boost the direction(s) that walk down the BFS distance to
    # the remembered enemy general.  Wrap artifacts from roll land only on
    # off-board (illegal) pairs, which the final legality mask kills.
    dist = planes[_P["dist_enemy_general"]]
    downhill = jnp.stack(
        [jnp.roll(dist, (-dr, -dc), (0, 1)) < dist - 1e-4
         for dr, dc in DIRS_RC], -1).reshape(N_CELLS, N_DIRS)
    dir_scores = dir_scores * (1.0 + S_HUNT * downhill)
    n_cast = planes[_P["own_castles"]].sum()
    build_score = jnp.where(n_cast >= BUILD_CAP, S_PASS, S_CITY)
    int_scores = jnp.concatenate([
        dir_scores,                                        # 0..3 all
        dir_scores * S_HALF,                               # 4..7 half, discounted
        jnp.full((N_CELLS, 1), 1.0) * build_score,         # 8 BUILD, soft-capped
        jnp.full((N_CELLS, 1), S_PASS),                    # 9 PASS, held down
    ], -1)

    kth = jnp.sort(own_army.reshape(N_CELLS))[-TOP_K]
    top = (own_army >= kth) & (own_army > 0)
    src_scores = (S_DEFAULT + S_TOPK * top).reshape(N_CELLS, 1)

    scores = src_scores * int_scores
    flat = jax.nn.log_softmax(
        jnp.where(legal, jnp.log(scores), NEG_INF).reshape(-1))
    return flat.reshape(N_CELLS, N_INT)


expander_prior_v = jax.vmap(expander_prior)
