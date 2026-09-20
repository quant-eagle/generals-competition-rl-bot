"""Q-boosting: the algebra, the degenerate case, and the variance claim.

The advantage estimator follows Fan & Farina (arXiv:2605.19235) in place of
GAE.  The trace is pinned three ways: against a hand-written GAE in the case
where the two must coincide, against a hand-computed two-step example, and
against the exact variance reduction the control-variate argument predicts.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax.random as jr

from train import net
from train.learn import LossCfg, danger_labels, qboost, top_advantage


def gae_ref(reward, value, terminal, done, gamma, lam):
    """Plain GAE, written out longhand as an independent reference."""
    T, N = reward.shape
    adv = np.zeros((T, N))
    nxt = np.zeros(N)
    v_next = np.asarray(value[-1])
    for t in range(T - 1, -1, -1):
        vn = np.where(terminal[t], 0.0, np.where(done[t], value[t], v_next))
        delta = reward[t] + gamma * vn - value[t]
        nxt = delta + gamma * lam * np.where(done[t], 0.0, nxt)
        adv[t] = nxt
        v_next = np.asarray(value[t])
    return adv


def _episode(T=12, N=3, seed=0):
    rng = np.random.default_rng(seed)
    reward = rng.normal(size=(T, N)).astype(np.float32)
    terminal = np.zeros((T, N), bool)
    terminal[-1] = True
    terminal[T // 2, 0] = True
    return reward, terminal, terminal.copy()


def test_qboost_reduces_to_gae_when_q_is_flat_in_the_action():
    """If Q(s,a) does not depend on a then Q = Vbar, and the trace must collapse
    to exactly GAE against that value function."""
    reward, terminal, done = _episode()
    v = np.random.default_rng(1).normal(size=reward.shape).astype(np.float32)
    adv, ret = qboost(jnp.asarray(reward), jnp.asarray(v), jnp.asarray(v),
                      jnp.asarray(terminal), jnp.asarray(done), 1.0, 0.95)
    want = gae_ref(reward, v, terminal, done, 1.0, 0.95)
    assert np.allclose(np.asarray(adv), want, atol=1e-5)
    assert np.allclose(np.asarray(ret), want + v, atol=1e-5), "ret = adv + Vbar"


def test_qboost_matches_a_hand_computed_two_step_episode():
    r = jnp.asarray([[0.0], [1.0]])
    q = jnp.asarray([[0.3], [0.8]])
    vb = jnp.asarray([[0.5], [0.6]])
    term = jnp.asarray([[False], [True]])
    g, lam = 1.0, 1.0
    adv, ret = qboost(r, q, vb, term, term, g, lam)

    d1 = 1.0 + 0.0 - 0.8                       # terminal: bootstrap 0
    d0 = 0.0 + 0.6 - 0.3
    assert np.isclose(float(adv[1, 0]), 0.8 - 0.6 + d1)
    assert np.isclose(float(adv[0, 0]), 0.3 - 0.5 + d0 + d1)
    assert np.isclose(float(ret[0, 0]), 0.3 + d0 + d1)


def test_the_control_variate_removes_downstream_action_noise():
    """The variance claim, made exact.

    A T-step chain of independent fair coin flips paying 0 or 1.  With a perfect
    critic, the Monte-Carlo advantage at t=0 has variance T/4: it carries the
    sampling noise of every future action.  Q-boosting subtracts Vbar - Q at
    each future step, whose expectation is zero, leaving variance 1/4: only the
    noise of the action being credited.
    """
    T, N = 16, 4000
    rng = np.random.default_rng(0)
    r = rng.integers(0, 2, size=(T, N)).astype(np.float32)
    # perfect critic for a uniform policy over {pay 0, pay 1}
    rem = (T - 1 - np.arange(T))[:, None] * 0.5
    q = r + rem                                  # Q(s_t, a_t)
    vb = np.broadcast_to(0.5 + rem, (T, N)).astype(np.float32)  # Vbar(s_t)
    term = np.zeros((T, N), bool); term[-1] = True

    adv, _ = qboost(*(jnp.asarray(x) for x in (r, q, vb, term, term)), 1.0, 1.0)
    boost_var = float(np.var(np.asarray(adv)[0]))
    mc_var = float(np.var(r.sum(0) - vb[0]))    # what GAE(lam=1) would carry

    assert np.isclose(boost_var, 0.25, atol=0.02), boost_var
    assert np.isclose(mc_var, T * 0.25, rtol=0.15), mc_var
    assert boost_var < mc_var / (T / 2), f"{boost_var:.3f} vs {mc_var:.3f}"


def test_a_draw_bootstraps_zero_and_is_never_a_loss():
    """The 1200-turn cap is terminal with reward 0."""
    r = jnp.zeros((3, 1))
    q = jnp.asarray([[0.4], [0.4], [0.4]])
    vb = jnp.asarray([[0.2], [0.2], [0.2]])
    term = jnp.asarray([[False], [False], [True]])
    adv, _ = qboost(r, q, vb, term, term, 1.0, 1.0)
    assert float(adv[2, 0]) == pytest.approx(0.4 - 0.2 + (0.0 - 0.4))
    assert float(adv[2, 0]) < 0.0  # a draw after over-valuing the state is bad
    assert not np.isnan(np.asarray(adv)).any()


def test_truncation_bootstraps_vbar_rather_than_scoring_a_loss():
    """A seed-horizon cut must not look like losing the game."""
    r = jnp.zeros((2, 2))
    q = jnp.full((2, 2), 0.5)
    vb = jnp.full((2, 2), 0.7)
    term = jnp.asarray([[False, False], [True, False]])
    done = jnp.asarray([[False, False], [True, True]])   # col 1 truncates
    adv, _ = qboost(r, q, vb, term, done, 1.0, 1.0)
    assert float(adv[1, 0]) == pytest.approx(0.5 - 0.7 + (0.0 - 0.5))
    assert float(adv[1, 1]) == pytest.approx(0.5 - 0.7 + (0.7 - 0.5))
    assert adv[1, 1] > adv[1, 0], "truncation must not be scored as a terminal"


def test_top_advantage_keeps_both_tails():
    """Magnitude, not signed value: dropping negatives is self-imitation."""
    adv = jnp.asarray([-5.0, 0.1, 0.0, 4.0, -0.2, 3.0, 0.05, -6.0])
    keep = np.asarray(top_advantage(adv, 0.5))
    assert sorted(keep.tolist()) == [0, 3, 5, 7]
    assert (np.asarray(adv)[keep] < 0).any() and (np.asarray(adv)[keep] > 0).any()


def test_top_advantage_fraction_matches_the_throughput_claim():
    """Keeping 25% of samples makes the update 4x cheaper; with the update at
    ~75% of an iteration that is a ~2.3x end-to-end speedup."""
    n = 65_536
    keep = top_advantage(jnp.asarray(np.random.default_rng(0).normal(size=n)),
                         LossCfg().adv_frac)
    assert keep.shape[0] == n // 4
    assert 0.25 + 0.75 / 4 == pytest.approx(1 / 2.2857, rel=1e-3)


def test_danger_labels_do_not_supervise_an_unobserved_tail():
    died = jnp.asarray([[0.0], [0.0], [1.0], [0.0], [0.0]])
    done = jnp.asarray([[False], [False], [True], [False], [False]])
    lab, valid = danger_labels(died, done, horizon=32)
    assert [float(x) for x in lab[:, 0]] == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert [float(x) for x in valid[:, 0]] == [1.0, 1.0, 1.0, 0.0, 0.0], \
        "the segment after the last death is unobserved and must not train"


def test_q_vbar_contracts_per_sample_not_across_the_batch():
    """The rank-r interaction term must be a per-sample dot product.  A plain
    matmul against (n, N_SRC, r) forms an (n, n, r) cross-sample outer product,
    which type-checks in some shapes and silently mixes states."""
    from train.learn import q_vbar
    rng = np.random.default_rng(0)
    n, S, I, r = 5, 7, 6, 3
    q0 = jnp.asarray(rng.normal(size=n).astype(np.float32))
    qs = jnp.asarray(rng.normal(size=(n, S)).astype(np.float32))
    qi = jnp.asarray(rng.normal(size=(n, I)).astype(np.float32))
    u = jnp.asarray(rng.normal(size=(n, S, r)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(n, I, r)).astype(np.float32))
    # the policy is a joint distribution over (cell, intent) and does not
    # factorise, so the test uses one that provably does not
    pj = rng.random((n, S, I)); pj /= pj.sum((1, 2), keepdims=True)

    got = np.asarray(q_vbar((q0, qs, qi, u, v),
                            jnp.asarray(pj.astype(np.float32))))
    for b in range(n):
        joint = (np.asarray(q0)[b] + np.asarray(qs)[b][:, None]
                 + np.asarray(qi)[b][None, :]
                 + np.asarray(u)[b] @ np.asarray(v)[b].T)
        assert np.isclose(got[b], (pj[b] * joint).sum(), atol=1e-5)


def test_gae_mode_runs_on_the_same_network_and_differs_from_qboost():
    """The estimator A/B must change the estimator and nothing else.

    `gae` takes Vbar -- which IS a state value, being sum_a pi(a) Q(s,a) -- so
    both modes share the network, the critic target and the whole pipeline.
    They must therefore be interchangeable in shape, and genuinely different in
    value whenever Q depends on the action.
    """
    from train.learn import gae
    reward, terminal, done = _episode(seed=3)
    rng = np.random.default_rng(7)
    vb = rng.normal(size=reward.shape).astype(np.float32)
    q = vb + rng.normal(size=reward.shape).astype(np.float32)   # Q != Vbar
    args = (jnp.asarray(reward), jnp.asarray(terminal), jnp.asarray(done))

    a_g, r_g = gae(args[0], jnp.asarray(vb), args[1], args[2], 1.0, 1.0)
    a_q, r_q = qboost(args[0], jnp.asarray(q), jnp.asarray(vb), args[1],
                      args[2], 1.0, 1.0)
    assert a_g.shape == a_q.shape and r_g.shape == r_q.shape
    assert not np.allclose(np.asarray(a_g), np.asarray(a_q)), \
        "the two estimators produced identical advantages"
    # ...and they must coincide exactly when Q carries no action information
    a_same, _ = qboost(args[0], jnp.asarray(vb), jnp.asarray(vb), args[1],
                       args[2], 1.0, 1.0)
    assert np.allclose(np.asarray(a_g), np.asarray(a_same), atol=1e-5)


def test_lambda_stops_mattering_when_the_critic_is_perfect():
    """The limiting property of Q-boosting.

    With an exact Q, delta+ = r + gamma*Vbar(s') - Q(s,a) vanishes identically,
    so the trace contributes nothing and the advantage collapses to Q - Vbar for
    every lam.  GAE has no equivalent: its delta stays non-zero because a state
    value cannot cancel the action that was taken.  With an imperfect critic
    lam trades bias against variance as usual (see the segment-length test).
    """
    T, N = 16, 3000
    rng = np.random.default_rng(1)
    r = rng.integers(0, 2, size=(T, N)).astype(np.float32)
    rem = (T - 1 - np.arange(T))[:, None] * 0.5
    q = r + rem                                   # exact Q for a uniform policy
    vb = np.broadcast_to(0.5 + rem, (T, N)).astype(np.float32)
    term = np.zeros((T, N), bool); term[-1] = True
    advs = [np.asarray(qboost(*(jnp.asarray(x) for x in (r, q, vb, term, term)),
                              1.0, lam)[0]) for lam in (0.5, 0.9, 1.0)]
    for a in advs[1:]:
        assert np.allclose(a, advs[0], atol=1e-5), "lam changed an exact-critic advantage"
    assert np.allclose(advs[0][0], r[0] - 0.5, atol=1e-5), "advantage should be Q - Vbar"


def test_explained_variance_is_the_vrpo_health_check():
    """VRPO's control variate only cancels action noise insofar as Q is right at
    the action taken.  q_ev is that, measured: 1.0 perfect, 0.0 no better than
    predicting the mean, negative worse than the mean.  If it sits near zero
    during a run, the estimator is adding variance and `--advantage gae` is the
    better choice."""
    rng = np.random.default_rng(0)
    ret = rng.normal(size=4096).astype(np.float32)
    ev = lambda pred: float(1.0 - np.var(ret - pred) / (np.var(ret) + 1e-8))
    assert ev(ret) > 0.999, "a perfect critic must score ~1"
    assert abs(ev(np.full_like(ret, ret.mean()))) < 0.01, "mean predictor ~0"
    assert ev(-ret) < 0, "an anti-correlated critic must score negative"


def test_qboost_trace_does_not_blow_up_at_the_shipped_segment_length():
    """With an imperfect critic the control-variate corrections no longer
    cancel, and at gamma=lam=1 nothing damps their sum, so the advantage grows
    with segment length.  GAE(1) telescopes and stays bounded; Q-boosting needs
    lam < 1 at a 512-tick segment."""
    from train.learn import gae
    from train.config import SEG
    rng = np.random.default_rng(0)
    T, N = SEG, 128
    r = np.zeros((T, N), np.float32); r[-1] = rng.choice([-1.0, 1.0], N)
    term = np.zeros((T, N), bool); term[-1] = True
    vb = rng.normal(0, 0.2, (T, N)).astype(np.float32)      # imperfect critic
    q = vb + rng.normal(0, 0.2, (T, N)).astype(np.float32)
    args = (jnp.asarray(r), jnp.asarray(q), jnp.asarray(vb),
            jnp.asarray(term), jnp.asarray(term))

    p99 = lambda a: float(np.percentile(np.abs(np.asarray(a)), 99))
    a1, _ = qboost(*args, 1.0, 1.0)
    a9, _ = qboost(*args, 1.0, LossCfg().lam)
    g1, _ = gae(args[0], args[2], args[3], args[4], 1.0, 1.0)

    assert p99(a1) > 4.0, "lam=1 should visibly accumulate at this length"
    assert p99(a9) < 2.0, "the shipped lam must keep the advantage bounded"
    assert p99(g1) < 2.0, "GAE(1) telescopes and stays bounded"
    assert LossCfg().lam < 1.0, "lam=1 is unsafe at the shipped segment length"


def _flat_for(cfg, n, seed=0):
    """A minimal `flat` dict: exactly the keys make_loss reads."""
    import jax.random as jr
    from train.config import (N_CELLS, N_FLOAT, N_INT, N_PACKED, N_PRIV, N_SCALAR,
                              N_SERIES, N_SRC, N_TSAMP, B)
    k = jr.split(jr.PRNGKey(seed), 12)
    return {
        "pbits": jr.randint(k[0], (n, N_PACKED, B, B), 0, 256).astype(jnp.uint8),
        "pfloat": jr.normal(k[1], (n, N_FLOAT, B, B), jnp.float16),
        "pscalar": jr.normal(k[2], (n, N_SCALAR), jnp.float16),
        "series": jr.normal(k[3], (n, N_SERIES, N_TSAMP), jnp.float16) * 0.3,
        "priv": jr.normal(k[4], (n, N_PRIV, B, B), jnp.float16),
        "legal": jnp.ones((n, N_SRC, N_INT), bool),
        "a_src": jr.randint(k[5], (n,), 0, N_SRC),
        "a_int": jr.randint(k[6], (n,), 0, N_INT),
        "logp": jr.normal(k[7], (n,)) * 0.1 - 2.0,
        "adv": jr.normal(k[8], (n,)),
        "ret": jr.normal(k[9], (n,)) * 0.5,
        "gen_target": jr.randint(k[10], (n,), 0, N_CELLS),
        "danger_label": (jr.uniform(k[11], (n,)) > 0.5).astype(jnp.float32),
        "danger_valid": jnp.ones((n,), jnp.float32),
    }


def test_gradient_accumulation_is_mathematically_identical():
    """Accumulation exists to cut backward activation memory.  It is safe only
    if it changes nothing about the optimisation: k micro-batches summed must
    give the same parameters as one full minibatch, to numerical noise."""
    import optax
    import train.learn as L
    from train.config import NetCfg

    cfg = NetCfg(dim=64, depth=1, heads=2, cell_dim=16, q_dim=32,
                 priv_hidden=16, temp_hidden=16)
    lc = L.LossCfg()
    flat = _flat_for(cfg, n=64)
    p0 = net.init(jr.PRNGKey(0), cfg)
    opt = optax.sgd(1e-2)                       # no state: isolates the gradient
    st = opt.init(p0)

    one, _, _ = L.make_update(cfg, lc, opt, 1, 1, accum=1)(
        p0, st, flat, jr.PRNGKey(1), 1.0)
    four, _, _ = L.make_update(cfg, lc, opt, 1, 1, accum=4)(
        p0, st, flat, jr.PRNGKey(1), 1.0)

    worst = max(float(jnp.abs(a - b).max()) for a, b in
                zip(jax.tree.leaves(one), jax.tree.leaves(four)))
    assert worst < 2e-5, f"accumulation changed the update by {worst:.2e}"


def test_data_parallel_pmean_matches_a_single_device():
    """Two devices with half the batch each must produce the parameters one
    device produces on the whole batch.  This holds only if gradients are
    averaged before the optimizer: optax is stateful, so per-device updates
    followed by parameter averaging would let the replicas' Adam moments
    diverge.  Runs on two simulated CPU devices."""
    import optax
    import train.learn as L
    from train.config import NetCfg

    if jax.device_count() < 2:
        pytest.skip("needs 2 devices: XLA_FLAGS=--xla_force_host_platform_device_count=2")

    cfg = NetCfg(dim=64, depth=1, heads=2, cell_dim=16, q_dim=32,
                 priv_hidden=16, temp_hidden=16)
    lc = L.LossCfg()
    flat = _flat_for(cfg, n=64)
    p0 = net.init(jr.PRNGKey(0), cfg)
    opt = optax.sgd(1e-2)
    st = opt.init(p0)

    single, _, _ = L.make_update(cfg, lc, opt, 1, 1)(
        p0, st, flat, jr.PRNGKey(1), 1.0)

    # split the batch across devices; every replica starts from the same params
    sharded = jax.tree.map(lambda x: x.reshape((2, 32) + x.shape[1:]), flat)
    rep = lambda t: jax.tree.map(lambda x: jnp.stack([x] * 2), t)
    par = jax.pmap(L.make_update(cfg, lc, opt, 1, 1, axis_name="d"),
                   axis_name="d", in_axes=(0, 0, 0, 0, None))
    multi, _, _ = par(rep(p0), rep(st), sharded, jr.split(jr.PRNGKey(1), 2), 1.0)

    # replicas must agree with each other...
    for leaf in jax.tree.leaves(multi):
        assert jnp.allclose(leaf[0], leaf[1], atol=1e-6), "replicas diverged"
    # ...and with the single-device result
    worst = max(float(jnp.abs(a[0] - b).max())
                for a, b in zip(jax.tree.leaves(multi), jax.tree.leaves(single)))
    assert worst < 2e-5, f"data-parallel update differs by {worst:.2e}"
