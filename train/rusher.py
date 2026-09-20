"""Rusher: a scripted opponent that plays the ladder's dominant kill pattern.

Strong ladder bots tend to win the same way: expand to parity, gather one big
stack, then commit it on a blind geodesic into the heart of enemy territory
while their own general stays hidden.  This agent scripts that shape so
evaluation includes the attack a policy most needs to survive.  It does not
need to be strong, only representative.

Phases (by obs.timestep):
  t < COMMIT_T   expand -- the largest small stack pokes toward the nearest
                 capturable cell, biased away from its own general; the
                 general garrisons and never moves.
  t >= COMMIT_T  commit -- once a >= COMMIT_ARMY stack exists, walk it toward
                 the densest visible enemy region, or the deepest reachable
                 fog while the enemy is unseen.  Capture the general on
                 adjacency.
"""
from functools import partial

import jax
import jax.numpy as jnp

from generals.core.observation import Observation

from generals.agents.agent import Agent
from generals.agents.hunter_agent import _bfs, _toward

COMMIT_T = 150        # ticks: expand before, hunt after
COMMIT_ARMY = 25      # stack size that arms the commit
HOME_R = 3            # stealth: no expansion churn within Manhattan 3 of home
GARRISON = 6          # army kept home; only the surplus is fed out


class RusherAgent(Agent):
    """Expand quietly, gather one big stack, march it at the enemy."""

    def __init__(self, id: str = "Rusher"):
        super().__init__(id)

    def reset(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        del key  # deterministic
        obs = observation
        a, mine = obs.armies, obs.owned_cells
        H, W = a.shape
        reach = jnp.int32(H * W)
        passable = ~(obs.mountains | obs.structures_in_fog
                     | (obs.castles & ~mine))
        mine_army = jnp.where(mine, a, 0)
        movable = mine & (a > 1)

        gen = mine & obs.generals
        g = jnp.argmax(gen.reshape(-1).astype(jnp.int32))
        gr, gc = g // W, g % W
        rr = jnp.arange(H)[:, None]
        cc = jnp.arange(W)[None, :]
        near_home = (jnp.abs(rr - gr) + jnp.abs(cc - gc)) <= HOME_R
        from_mine = _bfs(passable, mine)
        from_gen = _bfs(passable, gen)   # stable anchor for target choice

        # ---- commit target: seen enemy general > enemy land (the sweep
        # uncovers the general through vision) > deepest reachable fog
        # (the blind commit) > farthest open cell.
        enemy = obs.opponent_cells
        egen0 = enemy & obs.generals
        fog = obs.fog_cells & passable & (from_gen < reach)
        deepest = fog & (from_gen == jnp.max(jnp.where(fog, from_gen, -1)))
        open_ = passable & ~mine & (from_gen < reach)
        far_open = open_ & (from_gen == jnp.max(
            jnp.where(open_, from_gen, -1)))
        heart = jnp.where(jnp.any(egen0), egen0,
                jnp.where(jnp.any(enemy), enemy,
                jnp.where(jnp.any(fog), deepest, far_open)))

        to_heart = _bfs(passable, heart)
        h_dir, h_nbr = _toward(to_heart, passable)
        h_adv = (h_nbr < to_heart)

        # ---- expand goal: nearest capturable non-mine cell away from home
        growth = passable & ~mine & ~near_home & (from_mine < reach)
        growth = jnp.where(jnp.any(growth), growth, passable & ~mine)
        to_grow = _bfs(passable, growth)
        e_dir, e_nbr = _toward(to_grow, passable)
        e_adv = (e_nbr < to_grow)

        # ---- decapitate whenever adjacent with enough army (any phase)
        egen = enemy & obs.generals
        egen_army = jnp.sum(jnp.where(egen, a, 0))
        kill = jnp.any(egen) & movable & (to_heart == 1) & (a - 1 > egen_army)
        kill = kill & (_bfs(passable, egen) == 1)
        ki = jnp.argmax(jnp.where(kill, mine_army, -1).reshape(-1))

        # ---- commit: the hammer (largest non-general stack) walks the line
        hammer_ok = movable & ~gen & ~near_home
        hammer_pool = jnp.where(jnp.any(hammer_ok & h_adv),
                                hammer_ok & h_adv, movable & ~gen & h_adv)
        hi = jnp.argmax(jnp.where(hammer_pool, mine_army, -1).reshape(-1))
        max_stack = jnp.max(jnp.where(mine & ~gen, a, 0))
        committing = (obs.timestep >= COMMIT_T) & (max_stack >= COMMIT_ARMY)

        # ---- expand: largest small quiet stack that advances toward
        # growth. Big stacks (>= COMMIT_ARMY) are hammers, never pokes —
        # otherwise the odd-tick economy move yanks the hammer off its
        # march and it random-walks between the two goals.
        small = a < COMMIT_ARMY
        ex_ok = movable & ~gen & ~near_home & e_adv & small
        ex_pool = jnp.where(jnp.any(ex_ok), ex_ok,
                            movable & ~gen & e_adv & small)
        ex_pool = jnp.where(jnp.any(ex_pool), ex_pool,
                            movable & ~gen & e_adv)
        ei = jnp.argmax(jnp.where(ex_pool, mine_army, -1).reshape(-1))

        # ---- feed: the general's surplus seeds the map (hunter-style);
        # the garrison stays, so home vision leakage is minimal.
        gen_army = jnp.sum(jnp.where(gen, a, 0))
        gen_can = movable.reshape(-1)[g]
        feed = (gen_army >= 2 * GARRISON) & gen_can

        do_kill = jnp.any(kill)
        do_commit = ~do_kill & committing & jnp.any(hammer_pool)
        do_expand = ~do_kill & ~do_commit & jnp.any(ex_pool)
        do_feed = ~do_kill & ~do_commit & ~do_expand & feed
        i = jnp.where(do_kill, ki,
            jnp.where(do_commit, hi,
            jnp.where(do_expand, ei, g)))
        dirn = jnp.where(do_kill | do_commit,
                         h_dir.reshape(-1)[i], e_dir.reshape(-1)[i])
        idle = ~(do_kill | do_commit | do_expand | do_feed)
        return jnp.array([idle, i // W, i % W, dirn, do_feed],
                         dtype=jnp.int32)
