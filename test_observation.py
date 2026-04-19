import numpy as np
from env_internals import BoardsImplementation


def make_env(seed=42):
    return BoardsImplementation(4, 2, 1, 0, seed=seed, disable_sender=True,
                                receiver_goal_visibility_mode="full")


def test_each_guess_has_distinct_intensity_in_observation():
    """
    Guess i is encoded with intensity (i+1)*32 on channel 0, so swapping positions
    gives different observations. Channel count stays fixed at 3 regardless of
    n_landmarks, allowing weights to transfer across curriculum stages.
    """
    env = make_env()
    pos_a = (0, 1)
    pos_b = (3, 2)

    env.board2_guesses = [pos_a, pos_b]
    obs_ab = env.receiver_agent_view().copy()

    env.board2_guesses = [pos_b, pos_a]
    obs_ba = env.receiver_agent_view().copy()

    assert not np.array_equal(obs_ab, obs_ba), (
        "Swapping guess positions must produce different observations — "
        "each guess is encoded with a distinct intensity value"
    )


def test_channel_count_is_always_3():
    """Channel count stays fixed at 3 regardless of n_landmarks."""
    for n_landmarks in (1, 2, 3, 4, 6):
        env = BoardsImplementation(5, n_landmarks, 1, 0, seed=42)
        assert env.n_receiver_channels == 3
        assert env.n_sender_channels == 3
        assert env.receiver_agent_view().shape == (5, 5, 3)
        assert env.sender_agent_view().shape == (5, 5, 3)


def test_identical_observations_lead_to_different_action_outcomes():
    """
    Documents the ambiguity that was fixed: from what were identical observations,
    the same action moved different physical pixels.

    Action layout for 2 landmarks (n_landmarks=2):
      0          → do nothing
      1,2,3,4    → move guess[0] (up, down, left, right)
      5,6,7,8    → move guess[1] (up, down, left, right)
    """
    env = make_env()
    pos_a = (2, 2)
    pos_b = (0, 0)

    env.board2_guesses = [pos_a, pos_b]
    obs_before_ordering_a = env.receiver_agent_view().copy()
    env.receiver_agent_action(4)  # move guess[0] right
    outcome_a = env.board2_guesses[0]

    env.board2_guesses = [pos_b, pos_a]
    obs_before_ordering_b = env.receiver_agent_view().copy()
    env.receiver_agent_action(4)  # same action, now moves pos_b
    outcome_b = env.board2_guesses[0]

    # After the fix: observations are now different
    assert not np.array_equal(obs_before_ordering_a, obs_before_ordering_b)

    # Same action still moves different physical pixels (because starting positions differ)
    assert outcome_a != outcome_b
