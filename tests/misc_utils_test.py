import misc_utils

import numpy as np
import pytest


def test_compute_gae_shape_and_dtype():
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.5, 0.25, -1.0], dtype=np.float32)
    dones = np.array([0, 0, 1], dtype=np.int32)

    adv = misc_utils.compute_gae(
        rewards, values, dones,
        gamma=0.99, gae_lambda=0.95, last_value=0.0
    )

    assert adv.shape == rewards.shape
    assert adv.dtype == np.float32


def test_empty_input_returns_empty():
    rewards = np.array([], dtype=np.float32)
    values = np.array([], dtype=np.float32)
    dones = np.array([], dtype=np.int32)

    adv = misc_utils.compute_gae(
        rewards, values, dones,
        gamma=0.99, gae_lambda=0.95, last_value=0.0
    )

    assert adv.shape == (0,)
    assert adv.dtype == np.float32


def test_lambda_zero_equals_one_step_td_residuals():
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dones = np.array([0, 0, 0], dtype=np.int32)
    gamma = 0.9
    last_value = 0.7

    adv = misc_utils.compute_gae(
        rewards, values, dones,
        gamma=gamma, gae_lambda=0.0, last_value=last_value
    )

    expected = np.empty_like(rewards, dtype=np.float32)
    expected[0] = rewards[0] + gamma * values[1] - values[0]
    expected[1] = rewards[1] + gamma * values[2] - values[1]
    expected[2] = rewards[2] + gamma * last_value - values[2]

    np.testing.assert_allclose(adv, expected, rtol=0, atol=1e-6)


def test_done_cuts_bootstrap_and_accumulation():
    # Done at t = 1 should remove bootstrap from t = 1 (no `gamma * next_value` term) and prevent t = 0 advantage
    # from accumulating past t = 1.
    rewards = np.array([1.0, 10.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([0, 1, 0], dtype=np.int32)  # terminal at t=1
    gamma = 0.99
    lam = 0.95
    last_value = 123.0

    adv = misc_utils.compute_gae(
        rewards, values, dones,
        gamma=gamma, gae_lambda=lam, last_value=last_value
    )

    gae2 = rewards[2] + gamma * last_value - values[2]
    gae1 = rewards[1] - values[1]
    # Since gae_t1 corresponds to t = 1 which is terminal, and because dones[1] = 1, the accumulation into
    # t = 0 should not include beyond t = 1.
    gae0 = (rewards[0] + gamma * values[1] - values[0]) + gamma * lam * gae1
    expected = np.array([gae0, gae1, gae2], dtype=np.float32)
    np.testing.assert_allclose(adv, expected, rtol=0, atol=1e-6)


def test_matches_reference_implementation_random():
    rng = np.random.default_rng(0)
    t = 50
    rewards = rng.normal(size=t).astype(np.float32)
    values = rng.normal(size=t).astype(np.float32)
    dones = rng.integers(0, 2, size=t).astype(np.int32)
    gamma = 0.97
    lam = 0.91
    last_value = float(rng.normal())

    expected = np.zeros(t, dtype=np.float32)
    gae = 0.0
    for i in range(t - 1, -1, -1):
        nt = 1.0 - float(dones[i])
        nv = last_value if i == t - 1 else float(values[i + 1])
        delta = float(rewards[i]) + gamma * nv * nt - float(values[i])
        gae = delta + gamma * lam * nt * gae
        expected[i] = gae

    out = misc_utils.compute_gae(
        rewards, values, dones,
        gamma=gamma, gae_lambda=lam, last_value=last_value
    )

    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("dones_dtype", [np.bool_, np.int32, np.float32])
def test_dones_dtype_variants(dones_dtype):
    rewards = np.array([1.0, 2.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([0, 1], dtype=dones_dtype)
    adv = misc_utils.compute_gae(
        rewards, values, dones,
        gamma=0.9, gae_lambda=0.95, last_value=0.0
    )
    assert adv.shape == (2,)
