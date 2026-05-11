"""Navigation tests: verify each agent can learn to navigate independently.

Receiver navigation (disable_sender=True, receiver_goal_visibility_mode="full"):
  - Receiver sees all landmark positions and must navigate its guesses onto them.
  - Metric: performances_dist = 1 - current_dist / start_dist  (0=no improvement, 1=perfect)
  - n_landmarks=1: achieves ~0.92; threshold >= 0.70
  - n_landmarks=2: threshold >= 0.60  (mean ~0.62; the metric is capped below 0.70 because
    episodes that start with guesses already near landmarks have a tiny start_dist denominator
    and any stochastic mistake produces a large negative that offsets the 46% perfect
    episodes; fix_receiver_shaping_assignment=True + instant_mult=0 + shaping_mult=5 is the
    best configuration found: it eliminates Hungarian-flip gradient inconsistency and
    reduces noise from the board-change instant reward)

Sender navigation (disable_receiver=True):
  - Receiver is disabled (guesses frozen); sender moves clues toward landmarks.
  - The clue→landmark assignment is fixed at the start of each episode (locked
    assignment) so PBRS shaping gradients remain consistent throughout the episode
    and do not suffer from the Hungarian flip that affects naive optimal_distance
    shaping for n_landmarks ≥ 2.
  - instant_multiplier=0 isolates the PBRS shaping signal from the dominant
    board-change reward that otherwise masks the directional gradient.
  - Metric: clue_alignment_dist (total optimal distance, decreasing is better)
  - n_landmarks=1: improves by ~0.60; threshold >= 0.45
  - n_landmarks=2: improves by ~0.50 after 2000 episodes; threshold >= 0.40
"""
import math
import pytest
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from env_wrapper import BoardsWrapper
from env_internals import BoardsImplementation
from agent_architecture import AgentParams, GreedySenderAgent, PPOAgent, RandomAgent
from training_loop import train_agents


_SIZE = 4
_HISTORY_LEN = 4
_HIDDEN_SIZE = 128
_MAX_MOVES = _SIZE ** 2 * 4  # 64 total moves (32 per agent) per episode


def _make_env(
    n_landmarks,
    n_clues,
    disable_sender,
    disable_receiver,
    receiver_goal_visibility_mode,
    instant_mult,
    end_mult,
    shaping_mult,
    sender_shaping_mult,
    seed,
    fix_receiver_shaping_assignment=False,
    align_receiver_assignment_with_clues=False,
):
    return BoardsWrapper(
        BoardsImplementation(
            size=_SIZE,
            n_landmarks=n_landmarks,
            n_clues=n_clues,
            n_questions=0,
            seed=seed,
            disable_sender=disable_sender,
            disable_receiver=disable_receiver,
            receiver_goal_visibility_mode=receiver_goal_visibility_mode,
        ),
        max_moves=_MAX_MOVES,
        history_len=_HISTORY_LEN,
        instant_multiplier=instant_mult,
        end_multiplier=end_mult,
        shaping_multiplier=shaping_mult,
        sender_shaping_multiplier=sender_shaping_mult,
        fix_receiver_shaping_assignment=fix_receiver_shaping_assignment,
        align_receiver_assignment_with_clues=align_receiver_assignment_with_clues,
    )


def _ppo(n_actions, n_channels, alpha, n_epochs, seed):
    params = AgentParams(
        gamma=0.99,
        alpha=alpha,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=32,
        n_epochs=n_epochs,
        seed=seed,
    )
    return PPOAgent(
        board_size=_SIZE,
        history_len=_HISTORY_LEN,
        n_actions=n_actions,
        hidden_size=_HIDDEN_SIZE,
        device="cpu",
        params=params,
        n_channels_per_frame=n_channels,
    )


def _locked_assignment(clues, landmarks):
    """Return the Hungarian-optimal (row, col) pairs for clue→landmark assignment."""
    cost = np.array([
        [math.sqrt((c[0] - l[0]) ** 2 + (c[1] - l[1]) ** 2) for l in landmarks]
        for c in clues
    ])
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def _locked_dist(clues, landmarks, assignment):
    """Total distance using a fixed clue→landmark assignment."""
    return sum(
        math.sqrt((clues[r][0] - landmarks[c][0]) ** 2 + (clues[r][1] - landmarks[c][1]) ** 2)
        for r, c in assignment
    )


# ── receiver navigation ───────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("n_landmarks,n_episodes,threshold,instant_mult,shaping_mult", [
    (1,  500, 0.70, 2.0, 2.0),
    (2, 3000, 0.60, 0.0, 5.0),
])
def test_receiver_navigation(n_landmarks, n_episodes, threshold, instant_mult, shaping_mult):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    window = n_episodes // 4

    env = _make_env(
        n_landmarks=n_landmarks,
        n_clues=1,
        disable_sender=True,
        disable_receiver=False,
        receiver_goal_visibility_mode="full",
        instant_mult=instant_mult,
        end_mult=10.0,
        shaping_mult=shaping_mult,
        sender_shaping_mult=0.0,
        seed=42,
        fix_receiver_shaping_assignment=True,
    )
    sender = RandomAgent(list(range(env.sender_n_actions)), seed=1)
    receiver = _ppo(env.receiver_n_actions, env.receiver_n_channels,
                    alpha=3e-4, n_epochs=4, seed=2)

    stats = train_agents(env, sender, receiver, n_episodes, learn_interval=32)

    perfs = np.array(stats["performances_dist"])
    late = float(np.mean(perfs[-window:]))
    assert late >= threshold, (
        f"n_landmarks={n_landmarks}: late performances_dist={late:.3f} < {threshold}"
    )


# ── sender navigation ─────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("n_landmarks,n_episodes,min_dist_reduction", [
    (1, 1000, 0.45),
    (2, 2000, 0.40),
])
def test_sender_navigation(n_landmarks, n_episodes, min_dist_reduction):
    """Sender trains alone with locked-assignment PBRS shaping.

    The clue→landmark assignment is fixed once per episode at reset so that
    shaping gradients remain consistent throughout and do not suffer from the
    Hungarian-flip artefact.  We verify the mean total clue-to-landmark distance
    (Hungarian-optimal, evaluated at episode end) drops by at least
    min_dist_reduction from the early-training window to the late-training window.
    """
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    window = n_episodes // 4
    learn_interval = 16
    shaping_gamma = 0.99
    shaping_mult = 5.0

    # instant_mult=0 and end_mult=0: all reward comes from manual PBRS shaping.
    env = _make_env(
        n_landmarks=n_landmarks,
        n_clues=n_landmarks,
        disable_sender=False,
        disable_receiver=True,
        receiver_goal_visibility_mode="none",
        instant_mult=0.0,
        end_mult=0.0,
        shaping_mult=0.0,
        sender_shaping_mult=0.0,  # shaping applied manually via locked assignment
        seed=42,
    )
    sender = _ppo(env.sender_n_actions, env.sender_n_channels,
                  alpha=1e-3, n_epochs=8, seed=1)
    receiver = RandomAgent(list(range(env.receiver_n_actions)), seed=2)

    clue_dists = []
    for episode in range(n_episodes):
        env.reset()
        assignment = _locked_assignment(env.env.board1_clues, env.env.board1_landmarks)
        done = False

        while not done:
            s_obs = env.sender_observe()
            s_action, s_logp, s_val = sender.choose_action(s_obs)

            pre_d = _locked_dist(env.env.board1_clues, env.env.board1_landmarks, assignment)
            s_rew, done = env.sender_act(s_action)  # s_rew == 0 (all multipliers 0)
            post_d = _locked_dist(env.env.board1_clues, env.env.board1_landmarks, assignment)
            shaped_rew = (pre_d - shaping_gamma * post_d) * shaping_mult + s_rew

            if done:
                sender.remember(s_obs, s_action, s_logp, s_val,
                                shaped_rew + env.get_final_reward(), True)
                break
            r_obs = env.receiver_observe()
            r_rew, done = env.receiver_act(receiver.choose_action(r_obs)[0])
            if done:
                sender.remember(s_obs, s_action, s_logp, s_val,
                                shaped_rew + env.get_final_reward(), True)
            else:
                sender.remember(s_obs, s_action, s_logp, s_val, shaped_rew, False)

        clue_dists.append(env.get_clue_landmark_distance())

        if (episode + 1) % learn_interval == 0 or episode == n_episodes - 1:
            sender.learn()

    early = float(np.mean(clue_dists[:window]))
    late = float(np.mean(clue_dists[-window:]))
    assert early - late >= min_dist_reduction, (
        f"n_landmarks={n_landmarks}: clue_alignment_dist did not drop enough. "
        f"early={early:.3f}, late={late:.3f}, drop={early - late:.3f} < {min_dist_reduction}"
    )


# ── communication ─────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("n_landmarks,n_episodes,threshold,instant_mult,shaping_mult", [
    (1,  500, 0.50, 2.0, 2.0),
    (2, 6000, 0.10, 0.0, 5.0),
])
def test_communication(n_landmarks, n_episodes, threshold, instant_mult, shaping_mult):
    """Receiver learns to follow clue shadows from a scripted greedy sender.

    Sender (GreedySenderAgent) locks the clue→landmark assignment once per episode
    and greedily navigates clues toward their locked landmarks.  The receiver's PBRS
    shaping assignment is derived from the same clue→landmark assignment
    (align_receiver_assignment_with_clues=True), so shadow-i always guides guess-i:
    the visual signal and the reward gradient are consistent across every episode.

    Receiver has no direct landmark visibility (receiver_goal_visibility_mode="none")
    and must infer goal positions from clue shadows alone — the communication-following
    skill needed for full joint training.

    n_landmarks=1: one shadow, one guess; achieves ~0.50 mean; threshold >= 0.50.
    n_landmarks=2: two shadows, two guesses; consistent assignment eliminates
      cross-shadow ambiguity; learning is slow (warmup ~4500 ep) then jumps to
      ~0.17 mean; 6000 episodes needed; threshold >= 0.10.
    """
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    window = n_episodes // 4

    env = _make_env(
        n_landmarks=n_landmarks,
        n_clues=n_landmarks,
        disable_sender=False,
        disable_receiver=False,
        receiver_goal_visibility_mode="none",
        instant_mult=instant_mult,
        end_mult=10.0,
        shaping_mult=shaping_mult,
        sender_shaping_mult=0.0,
        seed=42,
        fix_receiver_shaping_assignment=True,
        align_receiver_assignment_with_clues=True,
    )
    sender = GreedySenderAgent(env)
    receiver = _ppo(env.receiver_n_actions, env.receiver_n_channels,
                    alpha=3e-4, n_epochs=4, seed=2)

    stats = train_agents(env, sender, receiver, n_episodes, learn_interval=32)

    perfs = np.array(stats["performances_dist"])
    late = float(np.mean(perfs[-window:]))
    assert late >= threshold, (
        f"n_landmarks={n_landmarks}: late performances_dist={late:.3f} < {threshold}"
    )
