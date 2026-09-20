"""Greedy evaluation: outcomes, play signature, and distance to a reference.

Evaluation is greedy (masked argmax), matching deployment exactly -- the shipped
policy takes an argmax, so a sampling eval would measure a bot that never ships.

Games run on the final curriculum stage, the competition's board distribution.
Per-tick scalars are recorded on device and every statistic is computed
afterwards on the host, which keeps the device code trivial and the statistics
easy to verify.

Besides win/loss, `profile` reports a play signature -- expansion and army
curves, castle timing and spacing, army concentration, time from sighting the
enemy general to the kill -- on the same axes as a reference signature measured
from top-ladder replays.  Mirror self-play win rate sits at 0.5 by
construction, so the signature is what shows whether play is getting better or
merely different.  It also reports where the policy puts its action mass
(builds, half-moves).  All of it is diagnostic; none of it enters the reward.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from generals.core.game import get_observation

from . import exec as X
from . import obs as O
from .config import B, DRAW_TURN, INT_BUILD, INT_HALF0, N_INT, NetCfg
from .net import act_logits_v
from .rollout import cstep_v, get_obs_v

# Reference play signature, measured from ~1000 top-ladder replays, and the
# spread across the top-5 ladder players, used to normalise the distance.
REFERENCE = {
    "len_win": 298.0, "t_to_kill": 3.0, "blind_kill": 0.58,
    "land100": 50.0, "land200": 67.0, "land300": 74.0,
    "army100": 108.0, "army200": 183.0,
    "cast200": 1.0, "build1": 124.0, "cast_d_gen": 7.0, "cast_d_pair": 7.0,
    "top1_200": 0.089, "top1_peak": 0.333,
    "spear100": 22.0, "spear200": 21.0, "spear_commit": 9.0,
    "idle": 0.018,
}
SPREAD = {
    "len_win": 100.0, "t_to_kill": 25.0, "blind_kill": 0.20,
    "land100": 3.0, "land200": 5.0, "land300": 9.0,
    "army100": 6.0, "army200": 17.0,
    "cast200": 0.5, "build1": 12.0, "cast_d_gen": 4.0, "cast_d_pair": 4.0,
    "top1_200": 0.010, "top1_peak": 0.08,
    "spear100": 2.0, "spear200": 1.0, "spear_commit": 1.0,
    "idle": 0.07,
}


class Trace(NamedTuple):
    alive: jnp.ndarray      # (T, N) episode still running before this tick
    winner: jnp.ndarray     # (T, N)
    land: jnp.ndarray
    army: jnp.ndarray
    top1: jnp.ndarray
    spear: jnp.ndarray
    castles: jnp.ndarray
    passed: jnp.ndarray
    seen_gen: jnp.ndarray   # learner has ever seen the enemy general
    built: jnp.ndarray      # intent was BUILD
    half: jnp.ndarray       # intent was a HALF move rather than all-but-one


def _seat_stats(state, seat):
    """Per-env scalars for `seat` from the full state."""
    mine = state.ownership[seat]
    armies = state.armies.astype(jnp.float32)
    own = jnp.where(mine, armies, 0.0)
    tot = jnp.maximum(own.sum(), 1.0)
    flat = jnp.argmax(own)
    r, c = flat // B, flat % B
    gr, gc = state.general_positions[1 - seat]
    return (mine.sum().astype(jnp.float32), own.sum(), own.max() / tot,
            (jnp.abs(r - gr) + jnp.abs(c - gc)).astype(jnp.float32),
            (state.castles & mine).sum().astype(jnp.float32))


_seat_stats_v = jax.vmap(_seat_stats, in_axes=(0, None))


def make_eval(cfg: NetCfg, opp_act=None):
    """Greedy self-play (or vs a scripted agent) returning per-tick traces."""

    def run(params, opp_params, state, n_ticks):
        def step(carry, _):
            state, m0, m1, r0, r1, done = carry
            o0, o1 = get_obs_v(state, 0), get_obs_v(state, 1)
            m0, m1 = O.update_v(m0, o0), O.update_v(m1, o1)
            r0, r1 = O.push_v(r0, o0), O.push_v(r1, o1)
            z0, z1 = O.series_of_v(r0), O.series_of_v(r1)
            x0 = O.encode_v(m0, o0)
            lg0 = O.legal_mask_v(o0)
            # greedy is a single argmax over the masked grid, and the grid is
            # exact, so the argmax is always executable
            lg = act_logits_v(params, x0, z0, lg0, cfg)
            flat = jnp.argmax(lg.reshape(lg.shape[0], -1), -1)
            a_s0, a_i0 = flat // N_INT, flat % N_INT
            act0 = X.execute_v(a_s0, a_i0, o0)
            if opp_act is None:
                x1 = O.encode_v(m1, o1)
                lg1 = O.legal_mask_v(o1)
                l1 = act_logits_v(opp_params, x1, z1, lg1, cfg)
                f1 = jnp.argmax(l1.reshape(l1.shape[0], -1), -1)
                act1 = X.execute_v(f1 // N_INT, f1 % N_INT, o1)
            else:
                act1 = jax.vmap(opp_act)(o1)

            land, army, top1, spear, cast = _seat_stats_v(state, 0)
            acts = jnp.stack([act0, act1], 1)
            nxt, info = cstep_v(state, acts)

            # The winner is read after the step: state.winner before stepping
            # never observes the result of the final tick.
            rec = Trace(~done, info.winner, land, army, top1, spear, cast,
                        act0[:, 0] == 1, m0.enemy_general.any((1, 2)),
                        a_i0 == INT_BUILD,
                        (a_i0 >= INT_HALF0) & (a_i0 < INT_BUILD))
            nxt_done = done | (info.winner >= 0) | (nxt.time >= DRAW_TURN)
            # finished episodes stop advancing, so later ticks cannot pollute
            # the statistics of games that already ended
            keep = lambda a, b: jnp.where(
                done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
            state = jax.tree.map(keep, nxt, state)
            return (state, m0, m1, r0, r1, nxt_done), rec

        n = state.time.shape[0]
        blank = jax.vmap(lambda _: O.init_memory())(jnp.arange(n))
        blank_r = jax.vmap(lambda _: O.init_rings())(jnp.arange(n))
        (final, *_), tr = jax.lax.scan(
            step, (state, blank, blank, blank_r, blank_r,
                   jnp.zeros(n, bool)), None, length=n_ticks)
        return tr, final

    return run


def scripted(name: str):
    """A fixed scripted opponent, as an `opp_act` for `make_eval`.

    The curriculum gate needs a fixed reference: self-play win rate sits near
    0.5 by construction, so it cannot say "I am competent at this difficulty",
    which is the question the gate asks.
    """
    from generals import agents as A
    cls = {"expander": A.ExpanderAgent, "hunter": A.HunterAgent}.get(name)
    if cls is None:
        from generals.agents.harvester_agent import HarvesterAgent
        from .rusher import RusherAgent
        cls = {"rusher": RusherAgent, "harvester": HarvesterAgent}[name]
    ag = cls()
    return lambda o: ag.act(o, jax.random.PRNGKey(0))


def castle_spacing(final) -> tuple[float, float]:
    """(median castle->own-general distance, median castle->castle distance).

    The crowding surcharge on a build is max(0, 14 - 2d), so 7 is exactly the
    surcharge-free radius, and the strongest ladder play sits right on it.
    This measures whether exposing the build price as a plane is enough for
    the policy to find that spacing without being rewarded for it.
    """
    own = np.asarray(final.ownership[:, 0])
    cast = np.asarray(final.castles) & own
    gp = np.asarray(final.general_positions)[:, 0]
    dg, dp = [], []
    for i in range(cast.shape[0]):
        rc = np.argwhere(cast[i])
        if not len(rc):
            continue
        dg.append(np.abs(rc - gp[i]).sum(1).min())
        if len(rc) > 1:
            d = np.abs(rc[:, None] - rc[None]).sum(-1).astype(float)
            np.fill_diagonal(d, np.inf)
            dp.append(d.min())
    return (float(np.median(dg)) if dg else float("nan"),
            float(np.median(dp)) if dp else float("nan"))


def profile(tr: Trace, final=None) -> dict:
    """Per-tick traces -> outcome rates and the play-signature columns."""
    alive = np.asarray(tr.alive)
    T, N = alive.shape
    end = alive.sum(0)                                  # ticks played per env
    win = np.asarray(tr.winner)[np.clip(end - 1, 0, T - 1), np.arange(N)]
    land, army = np.asarray(tr.land), np.asarray(tr.army)
    top1, spear = np.asarray(tr.top1), np.asarray(tr.spear)
    cast, passed = np.asarray(tr.castles), np.asarray(tr.passed)
    seen = np.asarray(tr.seen_gen)

    at = lambda a, t: a[min(t, T - 1)][end > t]
    won = win == 0
    lost = win == 1
    # A game with no winner that used the whole window is unfinished, not a
    # draw: a real draw only occurs at DRAW_TURN, which eval windows do not
    # reach.  Conflating the two would read "never loses" off a bot that never
    # closes.
    unfinished = (win < 0) & (end >= T)
    out: dict[str, float] = {
        "n": float(N), "winrate": float(won.mean()),
        "lossrate": float(lost.mean()),
        "unfinished": float(unfinished.mean()),
        "drawrate": float(((win < 0) & ~unfinished).mean()),
        "len_win": float(np.median(end[won])) if won.any() else np.nan,
        "land100": float(np.median(at(land, 100))) if (end > 100).any() else np.nan,
        "land200": float(np.median(at(land, 200))) if (end > 200).any() else np.nan,
        "land300": float(np.median(at(land, 300))) if (end > 300).any() else np.nan,
        "army100": float(np.median(at(army, 100))) if (end > 100).any() else np.nan,
        "army200": float(np.median(at(army, 200))) if (end > 200).any() else np.nan,
        "cast200": float(np.median(at(cast, 200))) if (end > 200).any() else np.nan,
        "top1_200": float(np.median(at(top1, 200))) if (end > 200).any() else np.nan,
        "spear100": float(np.median(at(spear, 100))) if (end > 100).any() else np.nan,
        "spear200": float(np.median(at(spear, 200))) if (end > 200).any() else np.nan,
    }

    # Peak concentration is measured over the last 40 alive ticks, matching
    # the reference's definition.  A max over the whole episode is meaningless:
    # at t=0 a player owns only its general, so top1 = largest cell / total
    # army = 1.0 by construction, and spear_commit -- the spearhead at the peak
    # tick -- would report the initial general-to-general distance.
    window = np.zeros_like(alive)
    for i in range(N):
        lo = max(int(end[i]) - 40, 0)
        window[lo:int(end[i]), i] = True
    window &= alive
    has = window.any(0)
    masked = np.where(window, top1, -1.0)
    peak_t = masked.argmax(0)
    out["top1_peak"] = float(np.median(masked.max(0)[has])) if has.any() else np.nan
    out["spear_commit"] = float(np.median(spear[peak_t, np.arange(N)][has])) \
        if has.any() else np.nan

    first_build = np.where(cast.max(0) > 0, (cast > 0).argmax(0), -1)
    out["build1"] = float(np.median(first_build[first_build > 0])) \
        if (first_build > 0).any() else np.nan
    out["idle"] = float((passed & alive).sum() / max(alive.sum(), 1))

    # Where the policy puts its action mass.  Diagnostics, never gates: the
    # fractions are path-dependent -- they measure what was learnable first as
    # much as what is best.
    live = max(alive.sum(), 1)
    out["build_frac"] = float((np.asarray(tr.built) & alive).sum() / live)
    out["half_frac"] = float((np.asarray(tr.half) & alive).sum() / live)

    ever = seen.max(0)
    first_seen = np.where(ever, seen.argmax(0), -1)
    ttk = np.where(ever, end - first_seen, np.nan)
    out["t_to_kill"] = float(np.nanmedian(ttk[won])) if won.any() else np.nan
    out["blind_kill"] = float(np.nanmean(ttk[won] <= 3)) if won.any() else np.nan

    # Unconditional versions.  `len_win` and `t_to_kill` are conditioned on
    # winning, so they are NaN exactly when it matters most whether games are
    # closing at all -- early in training.  These are readable throughout.
    decided = won | lost
    out["len_decided"] = float(np.median(end[decided])) if decided.any() else np.nan
    for t in (200, 300):
        out[f"decided_{t}"] = float((decided & (end <= t)).mean())
    # scouting, split from conversion: when is the enemy general first seen...
    out["t_seen"] = float(np.median(first_seen[ever])) if ever.any() else np.nan
    out["seen_frac"] = float(ever.mean())
    # ...and how long from sighting to the end, over every game that sighted it
    out["t_after_seen"] = float(np.nanmedian(ttk[ever])) if ever.any() else np.nan

    # Does the policy march?  Fraction of all episodes in which the largest
    # stack went from >=10 cells out to <=2 from the enemy general.  Taken over
    # all episodes, not wins: with generals >=17 apart winning requires
    # arriving, so conditioning on the outcome would make it true by
    # construction.
    far = (np.where(alive, spear, -1).max(0) >= 10)
    arrived = (np.where(alive, spear, 99).min(0) <= 2)
    out["g3_marches"] = float((far & arrived).mean())
    out["g3_marches_won"] = float((far & arrived)[won].mean()) if won.any() else 0.0
    out["spear_min"] = float(np.median(np.where(alive, spear, 99).min(0)))

    if final is not None:
        out["cast_d_gen"], out["cast_d_pair"] = castle_spacing(final)
    return out


def style_distance(prof: dict) -> tuple[float, int]:
    """(mean |ours - reference| / spread, number of axes it covers).

    The count matters: axes needing games that reach t=100/200/300 return NaN
    and drop out, so a distance from an agent whose games never resolve is
    averaged over a smaller, easier axis set than one whose games do.  Comparing
    two distances without their axis counts is comparing different metrics.
    Diagnostic only -- it never enters the reward.
    """
    ds = [abs(prof[k] - v) / SPREAD[k] for k, v in REFERENCE.items()
          if k in prof and prof[k] == prof[k]]
    return (float(np.mean(ds)) if ds else float("nan")), len(ds)


def report(prof: dict) -> str:
    lines = [f"{'axis':>14} {'ours':>9} {'ref':>9} {'|z|':>6}"]
    for k, v in REFERENCE.items():
        if k not in prof or prof[k] != prof[k]:
            continue
        lines.append(f"{k:>14} {prof[k]:>9.3f} {v:>9.3f} "
                     f"{abs(prof[k] - v) / SPREAD[k]:>6.2f}")
    d, n = style_distance(prof)
    lines.append(f"{'STYLE DIST':>14} {d:>9.3f}   over {n}/{len(REFERENCE)} axes")
    lines.append(f"{'winrate':>14} {prof['winrate']:>9.3f}   "
                 f"loss {prof['lossrate']:.3f}   draw {prof['drawrate']:.3f}   "
                 f"unfinished {prof['unfinished']:.3f}")
    lines.append(f"{'marches':>14} {prof['g3_marches']:>9.3f}   "
                 f"(diagnostic; of wins {prof['g3_marches_won']:.3f}, "
                 f"median closest approach {prof['spear_min']:.0f})")
    lines.append(f"{'closing':>14} decided {prof['len_decided']:>6.0f} ticks   "
                 f"by200 {prof['decided_200']:.2f}  by300 {prof['decided_300']:.2f}")
    lines.append(f"{'scouting':>14} seen {prof['seen_frac']:>6.2f} at t="
                 f"{prof['t_seen']:.0f}   then {prof['t_after_seen']:.0f} ticks "
                 f"to the end (reference: ~3)")
    lines.append(f"{'action mass':>14} build {prof['build_frac']:>6.3f}   "
                 f"half {prof['half_frac']:.3f}")
    return "\n".join(lines)
