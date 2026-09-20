"""VRPO learner: PPO with a Q-boosted advantage in place of GAE.

The estimator follows Fan & Farina (arXiv:2605.19235).  The rest of the recipe
-- sparse reward, top-advantage filtering, parameter EMA, annealed entropy --
follows Straka et al. (arXiv:2606.23348), with the entropy bonus extended to a
KL toward a scripted prior (magnet.py).

--- Q-boosting --------------------------------------------------------------

    Vbar(s)  = sum_a pi(a|s) Q(s,a)             # exact, no action sampling
    d+_t     = r_t + gamma*Vbar(s_{t+1}) - Q(s_t, a_t)
    A_t      = Q(s_t,a_t) - Vbar(s_t) + sum_k (gamma*lam)^k d+_{t+k}

Expanding the trace at lam=1 shows exactly what this buys:

    A_t = G_t - Vbar(s_t) + sum_{j>=1} gamma^j (Vbar(s_{t+j}) - Q(s_{t+j},a_{t+j}))

i.e. the Monte-Carlo advantage plus a per-step control variate.  Each correction
term has expectation zero under pi (E_a[Q(s,a)] = Vbar(s) by definition), so the
estimator stays unbiased while the action-sampling noise of every future step is
subtracted out.  GAE cannot do this: with a state-value critic there is nothing
to take the expectation over.  At lam=0 it collapses to the ordinary one-step TD
advantage against Vbar, so the two ends of the trace are both sane.

Vbar and Q(s,a) are computed in the rollout under the behaviour (tempered)
distribution and stored.  That is not a shortcut: d+ is an Expected-SARSA
backup for the policy that actually chose the actions, so the tempered
distribution is the correct one, and it keeps the update's critic loss to a
single scalar regression against `ret`.

--- why the factored Q is not a compromise ----------------------------------

Only the taken action is ever supervised, and the joint space is 441 x 10.  A
joint table would see each entry ~once per 200k samples.  Factored as
q0 + qs[i] + qi[j] + <u[i], v[j]>, one sample updates one source component and
one intent component, so all 451 components are supervised at the rate the
policy's own logits are.

--- reward ------------------------------------------------------------------

Terminal outcome only, gamma = 1.  No potential shaping: Straka et al. ablate
shaped against sparse and find shaping destabilises late training, Fan &
Farina train on terminal payoffs only, and a hand-written potential cannot
price army concentration anyway.  Hitting the turn cap is terminal with a zero
bootstrap; the rollout decides what a draw pays.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from . import magnet, obs
from .config import N_CELLS, N_INT, N_SRC, NetCfg
from .net import forward_v

DANGER_HORIZON = 32


class LossCfg(NamedTuple):
    advantage: str = "vrpo"      # "vrpo" (Q-boosting) or "gae"
    gamma: float = 1.0           # episodes are finite (turn cap), so gamma=1
    # lam = 0.9, not the 1.0 the unbiasedness argument suggests.  GAE at lam=1
    # telescopes -- sum_k gamma^k delta_k = G_t - V(s_t) -- so it is bounded by
    # the return however long the segment is.  Q-boosting does not:
    #
    #     A_t = G_t - Vbar_t + sum_{j>=1} gamma^j (Vbar_j - Q_j)
    #
    # Each correction has mean zero, but with an imperfect critic they do not
    # cancel, and at gamma=lam=1 nothing damps the sum: with a noisy critic the
    # p99 of |adv| grows ~3x between 64- and 512-tick segments, against returns
    # bounded in [-1, 1].  lam=0.9 keeps it flat, which is why long rollouts
    # pair with it.
    lam: float = 0.9
    clip: float = 0.2
    vf_coef: float = 0.5
    # Entropy is annealed as 0.05 * (t+1)^-0.2 with t in iterations, floored at
    # 0.001 (Straka et al.).  The exponent is tied to the run length: over
    # ~100k iterations it lands on a 0.005 endpoint, so a much shorter run
    # should rescale the exponent to keep that endpoint rather than copy 0.2.
    ent_coef: float = 0.05       # start of the schedule
    ent_power: float = 0.2
    ent_min: float = 0.001
    gen_coef: float = 0.1        # enemy-general location auxiliary
    danger_coef: float = 0.05    # death-within-32-ticks auxiliary
    max_grad_norm: float = 0.267   # Straka et al., Table V
    adv_frac: float = 0.25       # Table V: top-advantage filtering
    # Parameter EMA, applied once per iteration: a ~100-iteration horizon.  A
    # slower decay only makes sense for a run long enough to amortise it.
    ema_decay: float = 0.99


# --- Q head bookkeeping -----------------------------------------------------

def q_taken(qp, a_src, a_int):
    """Q(s, a) for the actions actually taken.  `qp` is net.q_parts output."""
    q0, qs, qi, u, v = qp
    n = jnp.arange(a_src.shape[0])
    return q0 + qs[n, a_src] + qi[n, a_int] + (u[n, a_src] * v[n, a_int]).sum(-1)


def q_vbar(qp, pi):
    """Vbar = sum_a pi(a|s) Q(s,a) over the joint, exact and still cheap.

    `pi` is (n, N_SRC, N_INT).  The policy is one categorical over the grid,
    but Q factorises, so the joint table is never materialised:

        Q(c,i) = q0 + qs[c] + qi[i] + <u[c], v[i]>
        Vbar   = q0 + <pc, qs> + <pv, qi> + sum_r (pi u_r) . v_r

    where pc and pv are pi's marginals.  Cost is O(N_SRC*N_INT*r) for the rank
    term, ~35k multiply-adds -- nothing against a trunk forward.

    Every leaf carries a leading sample axis, so the rank term contracts per
    sample; a plain matmul would form a cross-sample outer product.

    Masked actions carry pi = 0 (their logits are -1e30), so they drop out.
    """
    q0, qs, qi, u, v = qp
    pc, pv = pi.sum(-1), pi.sum(-2)                 # (n, N_SRC), (n, N_INT)
    # sum_{c,i} pi[c,i] <u[c], v[i]>, contracted per sample
    t = jnp.einsum("bci,bir->bcr", pi, v)
    rank = jnp.einsum("bcr,bcr->b", t, u)
    return q0 + (pc * qs).sum(-1) + (pv * qi).sum(-1) + rank


# --- advantage --------------------------------------------------------------

def qboost(reward, q_a, vbar, terminal, done, gamma, lam):
    """(T, N) -> (advantages, Q-targets), episode-boundary aware.

    `terminal` bootstraps 0 (the game really ended, draws included).
    `done & ~terminal` is a seed-horizon truncation and bootstraps Vbar(s_t) --
    one tick stale, which is a bias of one tick in a 40-120 tick horizon and far
    cheaper than the extra trunk forward the exact successor would cost.

    The Q-target is q_a + trace, which is the lam-return in Q-space; note
    adv = q_a - vbar + trace, so ret = adv + vbar exactly parallels GAE's
    ret = adv + value.
    """
    def body(carry, x):
        trace_next, vbar_next = carry
        r, q, vb, term, dn = x
        v_next = jnp.where(term, 0.0, jnp.where(dn, vb, vbar_next))
        delta = r + gamma * v_next - q
        trace = delta + gamma * lam * jnp.where(dn, 0.0, trace_next)
        return (trace, vb), (q - vb + trace, q + trace)

    zeros = jnp.zeros_like(vbar[0])
    _, (adv, ret) = jax.lax.scan(body, (zeros, vbar[-1]),
                                 (reward, q_a, vbar, terminal, done),
                                 reverse=True)
    return adv, ret


def gae(reward, value, terminal, done, gamma, lam):
    """Plain GAE, for A/B-ing the estimator.

    Deliberately runs on the same network: `value` is Vbar = sum_a pi(a) Q(s,a),
    which is a perfectly good state value, and the critic still regresses
    Q(s, a) to `ret`.  So switching estimators changes the estimator and nothing
    else, which is what makes the comparison mean anything.
    """
    def body(carry, x):
        adv_next, value_next = carry
        r, v, term, dn = x
        v_next = jnp.where(term, 0.0, jnp.where(dn, v, value_next))
        delta = r + gamma * v_next - v
        adv = delta + gamma * lam * jnp.where(dn, 0.0, adv_next)
        return (adv, v), adv

    zeros = jnp.zeros_like(value[0])
    _, adv = jax.lax.scan(body, (zeros, value[-1]),
                          (reward, value, terminal, done), reverse=True)
    return adv, adv + value


def danger_labels(died, done, horizon: int = DANGER_HORIZON):
    """(T, N) -> (label, valid) for the death-within-N auxiliary.

    Reverse scan for ticks-to-death within the episode.  `valid` masks the tail
    of the segment, where the outcome is simply not observed yet -- supervising
    it as "no death" would teach the head that late states are always safe.
    """
    BIG = 1e9

    def body(carry, x):
        ttd_next, seen_next = carry
        d, dn = x
        ttd = jnp.where(d, 0.0, jnp.where(dn, BIG, 1.0 + ttd_next))
        seen = d | dn | seen_next
        return (ttd, seen), (ttd, seen)

    n = died.shape[1]
    init = (jnp.full((n,), BIG), jnp.zeros((n,), bool))
    _, (ttd, seen) = jax.lax.scan(body, init, (died > 0.5, done), reverse=True)
    return (ttd <= horizon).astype(jnp.float32), (seen | (ttd <= horizon)
                                                  ).astype(jnp.float32)


# --- loss -------------------------------------------------------------------

def make_loss(cfg: NetCfg, lc: LossCfg):
    def loss_fn(params, mb, temp, ent_coef, mag_coef=None):
        """`temp` must match the temperature the rollout sampled at.

        The behaviour policy is softmax(logits / temp); recomputing logp from
        untempered logits makes the PPO ratio exp(new - old) compare two
        different distributions, so it is systematically wrong whenever T != 1.
        """
        # planes/priv stay fp16 all the way in: `net._cast` converts them to
        # the compute dtype, and f16 -> bf16 is bit-identical to going via f32.
        # Unpacked once per minibatch, not per use.
        planes = obs.unpack_v(mb["pbits"], mb["pfloat"], mb["pscalar"])
        logits, qp, gen, danger = forward_v(
            params, planes, mb["series"], mb["priv"], mb["legal"], cfg)
        nb = logits.shape[0]
        # One categorical over (cell, intent): the log-prob is a single
        # lookup into a single masked log-softmax, so there is one PPO ratio.
        lp = jax.nn.log_softmax((logits / temp).reshape(nb, -1))
        n = jnp.arange(nb)
        a_flat = mb["a_src"] * N_INT + mb["a_int"]
        logp = lp[n, a_flat]

        adv = mb["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = jnp.exp(logp - mb["logp"])
        pg = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - lc.clip, 1 + lc.clip) * adv).mean()

        q_pred = q_taken(qp, mb["a_src"], mb["a_int"])
        v_loss = 0.5 * jnp.square(q_pred - mb["ret"]).mean()
        # Explained variance of the critic -- the key diagnostic for VRPO.
        # The control variate only cancels action noise to the extent that Q is
        # right at the action taken, so if q_ev stays near 0 the estimator is
        # adding variance rather than removing it, and `--advantage gae` is the
        # answer.  1.0 is perfect, 0.0 is "no better than predicting the mean".
        q_ev = 1.0 - jnp.var(mb["ret"] - q_pred) / (jnp.var(mb["ret"]) + 1e-8)

        # entropy over legal actions only, for logging: masked logits are -1e30
        # so their probability is 0 and they contribute nothing to the sum
        m_flat = mb["legal"].reshape(nb, -1)
        ent = -jnp.sum(jnp.where(m_flat, jnp.exp(lp) * lp, 0.0), -1).mean()

        gen_loss = optax.softmax_cross_entropy_with_integer_labels(
            gen, mb["gen_target"]).mean()
        d_lab, d_valid = mb["danger_label"], mb["danger_valid"]
        d_loss = (optax.sigmoid_binary_cross_entropy(danger, d_lab) * d_valid
                  ).sum() / jnp.maximum(d_valid.sum(), 1.0)

        # Regulariser: KL(pi || m) = -H(pi) + CE(pi, m) toward the scripted
        # expander prior m (magnet.py).  The -H half is the usual entropy
        # bonus; the CE half anchors a newborn policy to sane expansion.
        # Under one coefficient, sustaining exploration would also keep
        # re-strengthening a prior the policy has outgrown, so the halves are
        # decoupled: entropy keeps its slow schedule, the magnet gets its own
        # coefficient that train.py anneals to zero.  mag_coef=None ties the
        # two together again (the fused KL).
        lm = magnet.expander_prior_v(planes.astype(jnp.float32),
                                     mb["legal"]).reshape(nb, -1)
        p_flat = jnp.exp(lp)
        neg_h = jnp.sum(jnp.where(m_flat, p_flat * lp, 0.0), -1).mean()
        ce_m = jnp.sum(jnp.where(m_flat, -p_flat * lm, 0.0), -1).mean()
        kl_ref = neg_h + ce_m
        mc = ent_coef if mag_coef is None else mag_coef
        loss = (pg + lc.vf_coef * v_loss + ent_coef * neg_h + mc * ce_m
                + lc.gen_coef * gen_loss + lc.danger_coef * d_loss)
        return loss, {"pg": pg, "v": v_loss, "q_ev": q_ev,
                      "ent": ent, "kl_ref": kl_ref,
                      "gen": gen_loss, "danger": d_loss,
                      "kl": jnp.mean(mb["logp"] - logp),
                      "clipfrac": jnp.mean(
                          (jnp.abs(ratio - 1.0) > lc.clip).astype(jnp.float32)),
                      "adv_std": mb["adv"].std()}
    return loss_fn


def top_advantage(adv_flat, frac: float):
    """Indices of the top-|advantage| quartile (Straka et al.'s filtering).

    `lax.top_k` rather than a full argsort: we need the top quarter, not a total
    order, and one fused selection beats sorting 65,536 elements to discard
    three quarters of them.

    Magnitude, not signed value: PPO's gradient scales with the advantage, so
    the smallest-|A| samples are the ones contributing least, and dropping them
    keeps both "that was good" and "that was bad" signal.  Filtering by signed
    advantage would keep only successes, which is self-imitation -- a different
    algorithm with a different bias.

    This is a biased subsample of the gradient; Straka et al. report it wins
    anyway on both wall-clock and sample efficiency.
    """
    k = max(1, int(round(adv_flat.shape[0] * frac)))
    return jax.lax.top_k(jnp.abs(adv_flat), k)[1]


def make_prepare(lc: LossCfg):
    """Advantages, auxiliary labels, flattening and filtering, in one jit.

    As separate eager calls this stage is a scan, a dozen reshapes, a selection
    and twelve gathers, each its own dispatch (~8% of an iteration).  Jitted,
    XLA fuses the gathers into the reshapes and the stage is one launch.
    """
    @jax.jit
    def prepare(batch):
        if lc.advantage == "gae":
            adv, ret = gae(batch.reward, batch.vbar, batch.terminal,
                           batch.done, lc.gamma, lc.lam)
        else:
            adv, ret = qboost(batch.reward, batch.q_a, batch.vbar,
                              batch.terminal, batch.done, lc.gamma, lc.lam)
        d_lab, d_val = danger_labels(batch.died, batch.done)
        # League masking: seat-1 lanes of scripted-opponent envs carry
        # actions a script chose, so PPO's ratio is undefined on them.  A
        # zeroed advantage puts them at the bottom of the |adv| top-k, and the
        # filter never selects one (invalid lanes are a smaller share of the
        # batch than the 75% the filter drops).  Zero, not -inf: the filter
        # ranks |adv|.
        adv = adv * batch.valid
        flat = flatten_batch(batch, adv, ret, d_lab, d_val)
        keep = top_advantage(flat["adv"], lc.adv_frac)
        return jax.tree.map(lambda x: x[keep], flat)
    return prepare


def make_update(cfg: NetCfg, lc: LossCfg, opt, n_epochs: int, n_minibatch: int,
                accum: int = 1, axis_name: str | None = None):
    """-> jitted (params, opt_state, flat, key, temp) -> (params, state, stats).

    `flat` arrives already filtered (train.py does it, so the reference forward
    only pays for the samples that are actually trained on).
    """
    loss_fn = make_loss(cfg, lc)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update(params, opt_state, flat, key, temp=1.0, ent_coef=0.05,
               mag_coef=None):
        n = flat["adv"].shape[0]
        mb_size = n // n_minibatch

        def epoch(carry, ekey):
            params, opt_state = carry
            perm = jax.random.permutation(ekey, n)

            def minibatch(carry, i):
                params, opt_state = carry
                sl = jax.lax.dynamic_slice(perm, (i * mb_size,), (mb_size,))
                mb = jax.tree.map(lambda x: x[sl], flat)

                if accum == 1:
                    (_, stats), g = grad_fn(params, mb, temp, ent_coef,
                                            mag_coef)
                else:
                    # Gradient accumulation.  Backward activations, not the
                    # trajectory buffer, are what cap the minibatch on a 32 GB
                    # card.  Splitting it into `accum` micro-batches and
                    # averaging their gradients is mathematically identical --
                    # same samples, same effective batch, same number of
                    # optimizer steps -- at 1/accum the activation memory, and
                    # unlike rematerialisation it costs no extra compute.
                    micro = mb_size // accum
                    zero = jax.tree.map(jnp.zeros_like, params)

                    def acc(g_sum, j):
                        sub = jax.tree.map(
                            lambda x: jax.lax.dynamic_slice_in_dim(
                                x, j * micro, micro), mb)
                        (_, st), g = grad_fn(params, sub, temp, ent_coef,
                                             mag_coef)
                        return jax.tree.map(jnp.add, g_sum, g), st

                    g, stats = jax.lax.scan(acc, zero, jnp.arange(accum))
                    g = jax.tree.map(lambda x: x / accum, g)
                    stats = jax.tree.map(lambda x: x.mean(0), stats)

                if axis_name is not None:
                    # Data parallel.  Average gradients across devices before
                    # the optimizer, not parameters after it: optax is
                    # stateful, so per-device updates would let each replica's
                    # Adam moments diverge.  Averaging the gradient keeps every
                    # replica bit-identical by construction, which is also what
                    # makes checkpointing from device 0 correct.
                    g = jax.lax.pmean(g, axis_name)
                    stats = jax.lax.pmean(stats, axis_name)
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), stats

            return jax.lax.scan(minibatch, (params, opt_state),
                                jnp.arange(n_minibatch))

        (params, opt_state), stats = jax.lax.scan(
            epoch, (params, opt_state), jax.random.split(key, n_epochs))
        return params, opt_state, jax.tree.map(jnp.mean, stats)

    return update if axis_name is not None else jax.jit(update)


def ema_update(ema, params, decay: float):
    """Parameter EMA for evaluation and deployment (Straka et al. report
    roughly +30 Elo from playing the EMA).

    Applied once per iteration, not per minibatch: a per-minibatch decay of 0.99
    would average over a fraction of one iteration and mean nothing.
    """
    return jax.tree.map(lambda e, p: decay * e + (1 - decay) * p, ema, params)


def entropy_coef(lc: "LossCfg", it: int) -> float:
    """0.05 * (t+1)^-0.2 with t in iterations, floored at 0.001."""
    return max(lc.ent_min, lc.ent_coef * (it + 1) ** -lc.ent_power)


def power_law_schedule(lr_max: float, num: float = 0.5, power: float = 1.1,
                       lr_min: float = 5e-6, steps_per_iter: int = 1):
    """clip(num * (t+1)^-power, lr_min, lr_max), Straka et al. Table V, with
    t in iterations -- the same clock the entropy schedule runs on.

    optax feeds a schedule the optimizer-step count (one per minibatch), so
    `steps_per_iter` converts; without it the schedule would hit its floor
    within the first hundred iterations.

    Power law rather than cosine: correct at any stopping point.
    """
    def sched(step):
        it = step / float(max(steps_per_iter, 1))
        return jnp.clip(num * (it + 1.0) ** -power, lr_min, lr_max)
    return sched


def make_optimizer(lr, max_grad_norm: float, total_steps: int = 0,
                   power: float = 1.1, steps_per_iter: int = 1,
                   lr_scale: float = 1.0):
    """`lr_scale` multiplies the schedule output, floor included, so a resumed
    run can train hotter without re-deriving the schedule's clock."""
    sched = power_law_schedule(lr, power=power, steps_per_iter=steps_per_iter)
    scaled = (sched if lr_scale == 1.0
              else (lambda count: lr_scale * sched(count)))
    return optax.chain(optax.clip_by_global_norm(max_grad_norm),
                       optax.adam(scaled))


def flatten_batch(batch, adv, ret, d_label, d_valid) -> dict:
    """(T, N, ...) -> (T*N, ...) for minibatching."""
    f = lambda x: x.reshape((-1,) + x.shape[2:])
    return {"pbits": f(batch.pbits), "pfloat": f(batch.pfloat),
            "pscalar": f(batch.pscalar), "series": f(batch.series),
            "priv": f(batch.priv),
            "legal": f(batch.legal),
            "a_src": f(batch.a_src), "a_int": f(batch.a_int),
            "logp": f(batch.logp), "adv": f(adv), "ret": f(ret),
            "gen_target": f(batch.gen_target).astype(jnp.int32),
            "danger_label": f(d_label), "danger_valid": f(d_valid)}
