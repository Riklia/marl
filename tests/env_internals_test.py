from env_internals import BoardsImplementation

import env_test_helpers
import numpy as np
import pytest


def test_init_validation():
    with pytest.raises(ValueError):
        BoardsImplementation(size=0, n_landmarks=1, n_clues=1, n_questions=0)
    with pytest.raises(ValueError):
        BoardsImplementation(size=3, n_landmarks=0, n_clues=1, n_questions=0)
    with pytest.raises(ValueError):
        BoardsImplementation(size=3, n_landmarks=1, n_clues=0, n_questions=0)
    with pytest.raises(ValueError):
        BoardsImplementation(size=3, n_landmarks=1, n_clues=1, n_questions=-1)
    with pytest.raises(ValueError):
        BoardsImplementation(size=2, n_landmarks=2, n_clues=2, n_questions=0)
    with pytest.raises(ValueError):
        BoardsImplementation(size=3, n_landmarks=1, n_clues=1, n_questions=0, seed=-5)


def test_action_space_lengths():
    env = BoardsImplementation(
        size=4,
        n_landmarks=2,
        n_clues=1,
        n_questions=1,
        seed=42
    )
    assert len(env.sender_agent_actions) == 1 + 4 * env.n_clues
    assert len(env.receiver_agent_actions) == 1 + 4 * env.n_landmarks + 4 * env.n_questions


def test_populate_boards_invariants():
    env = BoardsImplementation(
        size=4,
        n_landmarks=2,
        n_clues=1,
        n_questions=1,
        seed=42
    )
    env.populate_boards()

    assert env_test_helpers.all_in_bounds(env.board1_landmarks, env.size)
    assert env_test_helpers.all_in_bounds(env.board1_clues, env.size)
    assert env_test_helpers.all_in_bounds(env.board2_guesses, env.size)
    assert env_test_helpers.all_in_bounds(env.board2_questions, env.size)

    assert env_test_helpers.no_self_overlap(env.board1_landmarks)
    assert env_test_helpers.no_self_overlap(env.board1_clues)
    assert env_test_helpers.no_self_overlap(env.board2_guesses)
    assert env_test_helpers.no_self_overlap(env.board2_questions)

    assert env_test_helpers.no_overlap(env.board1_landmarks, env.board1_clues)
    assert env_test_helpers.no_overlap(env.board2_guesses, env.board2_questions)


@pytest.mark.parametrize("linked_shadows", [True, False])
def test_sender_view_shadow_logic(linked_shadows):
    env = BoardsImplementation(
        size=4,
        n_landmarks=1,
        n_clues=1,
        n_questions=1,
        linked_shadows=linked_shadows,
        seed=42,
    )

    env.board1_landmarks = [(0, 0)]
    env.board1_clues = [(1, 0)]
    env.board2_questions = [(2, 0)]
    env.board1_q_shadows = [(3, 3)]

    img = env.sender_agent_view()
    assert img.shape == (4, 4, 3)

    assert env_test_helpers.pixel_equal(img[0, 0], (255, 0, 0))
    assert env_test_helpers.pixel_equal(img[0, 1], (0, 255, 0))
    if linked_shadows:
        # Blue comes from board2_questions
        assert env_test_helpers.pixel_equal(img[0, 2], (0, 0, 255))
        assert env_test_helpers.pixel_equal(img[3, 3], (0, 0, 0))
    else:
        # Blue comes from board1_q_shadows
        assert env_test_helpers.pixel_equal(img[3, 3], (0, 0, 255))
        assert env_test_helpers.pixel_equal(img[0, 2], (0, 0, 0))


@pytest.mark.parametrize("linked_shadows", [True, False])
def test_receiver_view_shadow_logic(linked_shadows):
    env = BoardsImplementation(
        size=4,
        n_landmarks=1,
        n_clues=1,
        n_questions=1,
        linked_shadows=linked_shadows,
        seed=42,
    )

    env.board2_guesses = [(0, 0)]
    env.board2_questions = [(1, 0)]
    env.board1_clues = [(2, 0)]
    env.board2_c_shadows = [(3, 3)]

    img = env.receiver_agent_view()
    assert img.shape == (4, 4, 3)

    assert env_test_helpers.pixel_equal(img[0, 0], (255, 0, 0))
    assert env_test_helpers.pixel_equal(img[0, 1], (0, 255, 0))
    if linked_shadows:
        # Blue from board1_clues
        assert env_test_helpers.pixel_equal(img[0, 2], (0, 0, 255))
        assert env_test_helpers.pixel_equal(img[3, 3], (0, 0, 0))
    else:
        # Blue from board2_c_shadows
        assert env_test_helpers.pixel_equal(img[3, 3], (0, 0, 255))
        assert env_test_helpers.pixel_equal(img[0, 2], (0, 0, 0))


def test_sender_action_do_nothing_sets_useless_flag():
    env = BoardsImplementation(
        size=4,
        n_landmarks=1,
        n_clues=1,
        n_questions=1,
        seed=42,
    )
    before = env.board1_clues.copy()
    env.sender_agent_action(0)
    assert env.useless_action_flag is True
    assert env.board1_clues == before


def test_sender_action_out_of_bounds_is_useless():
    env = BoardsImplementation(
        size=3,
        n_landmarks=1,
        n_clues=1,
        n_questions=0,
        seed=42,
    )

    # Put clue at top row so "up" goes out of bounds.
    env.board1_landmarks = [(2, 2)]
    env.board1_clues = [(1, 0)]
    before = env.board1_clues.copy()

    # Action index 1 is ("clue", 0, up) per construction
    env.sender_agent_action(1)
    assert env.useless_action_flag is True
    assert env.board1_clues == before


def test_sender_action_collision_is_useless():
    env = BoardsImplementation(
        size=3,
        n_landmarks=1,
        n_clues=1,
        n_questions=0,
        seed=42,
    )

    # Move clue right into landmark
    env.board1_clues = [(0, 0)]
    env.board1_landmarks = [(1, 0)]
    before = env.board1_clues.copy()

    # Right is index 4 (None=0, up=1, down=2, left=3, right=4)
    env.sender_agent_action(4)
    assert env.useless_action_flag is True
    assert env.board1_clues == before


def test_sender_action_valid_move_updates_position():
    env = BoardsImplementation(
        size=3,
        n_landmarks=1,
        n_clues=1,
        n_questions=0,
        seed=42,
    )

    env.board1_landmarks = [(2, 2)]
    env.board1_clues = [(0, 0)]

    # Right is index 4 (None=0, up=1, down=2, left=3, right=4)
    env.sender_agent_action(4)

    assert env.useless_action_flag is False
    assert env.board1_clues == [(1, 0)]


def test_receiver_action_guess_valid_move_updates_position():
    env = BoardsImplementation(
        size=3,
        n_landmarks=1,
        n_clues=1,
        n_questions=1,
        seed=42,
    )

    env.board2_guesses = [(0, 0)]
    env.board2_questions = [(2, 2)]

    # Right is index 4 (None=0, up=1, down=2, left=3, right=4)
    env.receiver_agent_action(4)

    assert env.useless_action_flag is False
    assert env.board2_guesses == [(1, 0)]


def test_sender_view_does_not_observe_receiver_guesses():
    env = BoardsImplementation(
        size=4,
        n_landmarks=1,
        n_clues=1,
        n_questions=1,
        linked_shadows=True,
        seed=42,
    )

    # Sender-visible state
    env.board1_landmarks = [(0, 0)]
    env.board1_clues = [(1, 0)]
    env.board2_questions = [(2, 0)]

    # Receiver-only state, which is not visible for the sender.
    env.board2_guesses = [(3, 0)]
    img1 = env.sender_agent_view().copy()

    # Change the receiver guess, which also should not affect the sender's view.
    env.board2_guesses = [(0, 3)]
    img2 = env.sender_agent_view().copy()

    assert np.array_equal(img1, img2)


def test_receiver_view_does_not_observe_landmarks():
    env = BoardsImplementation(
        size=4,
        n_landmarks=1,
        n_clues=1,
        n_questions=1,
        linked_shadows=True,
        seed=42,
    )

    # Receiver-visible state
    env.board2_guesses = [(0, 0)]
    env.board2_questions = [(1, 0)]
    env.board1_clues = [(2, 0)]

    # Sender-only state, which is not visible for sender.
    env.board1_landmarks = [(3, 0)]
    img1 = env.receiver_agent_view().copy()

    # Change the landmark position, which also should not affect the sender's view.
    env.board1_landmarks = [(0, 3)]
    img2 = env.receiver_agent_view().copy()

    assert np.array_equal(img1, img2)

def test_reward_function_perfect_alignment_is_one():
    env = BoardsImplementation(size=4, n_landmarks=1, n_clues=1, n_questions=0, seed=42)

    # Force known state (distance = 0)
    env.board1_landmarks = [(0, 0)]
    env.board2_guesses = [(0, 0)]

    # Make denominators deterministic
    env.neutral_distance = 10.0
    env.start_distance = 5.0

    reward, performance = env.reward_function()

    assert reward == 1.0
    assert performance == 1.0


def test_reward_function_is_zero_when_distance_equals_neutral_distance():
    env = BoardsImplementation(size=4, n_landmarks=1, n_clues=1, n_questions=0, seed=42)

    # With the test stubbed greedy_distance: Manhattan distance between (0,0) and (3,0) = 3
    env.board1_landmarks = [(0, 0)]
    env.board2_guesses = [(3, 0)]

    env.neutral_distance = 3.0
    env.start_distance = 6.0

    reward, performance = env.reward_function()
    assert reward == 0.0
    assert performance == 1.0 - 3.0 / 6.0


def test_reward_function_decreases_with_distance():
    env = BoardsImplementation(
        size=5,
        n_landmarks=1,
        n_clues=1,
        n_questions=0,
        seed=42
    )

    env.neutral_distance = 10.0
    env.start_distance = 10.0
    env.board1_landmarks = [(0, 0)]

    env.board2_guesses = [(1, 0)]
    r_close, p_close = env.reward_function()

    env.board2_guesses = [(4, 0)]
    r_far, p_far = env.reward_function()

    assert r_close > r_far
    assert p_close > p_far


def test_reward_function_performance_is_negative_when_worse_than_start():
    env = BoardsImplementation(
        size=5,
        n_landmarks=1,
        n_clues=1,
        n_questions=0,
        seed=42
    )

    env.board1_landmarks = [(0, 0)]
    env.board2_guesses = [(4, 0)]

    env.neutral_distance = 10.0
    env.start_distance = 2.0

    reward, performance = env.reward_function()

    assert reward == 1.0 - 4.0 / 10.0
    assert performance == 1.0 - 4.0 / 2.0

def test_env_deterministic_given_seed_and_actions():
    from env_internals import BoardsImplementation

    env1 = BoardsImplementation(size=5, n_landmarks=2, n_clues=2, n_questions=1, linked_shadows=True, seed=123)
    env2 = BoardsImplementation(size=5, n_landmarks=2, n_clues=2, n_questions=1, linked_shadows=True, seed=123)

    env1.populate_boards()
    env2.populate_boards()

    actions = [0, 1, 4, 0, 2, 7]

    for a in actions:
        env1.sender_agent_action(a)
        env2.sender_agent_action(a)

        assert env1.board1_clues == env2.board1_clues
        assert env1.useless_action_flag == env2.useless_action_flag
        assert np.array_equal(env1.sender_agent_view(), env2.sender_agent_view())

def test_sender_action_preserves_clue_identity_order():
    env = BoardsImplementation(size=5, n_landmarks=1, n_clues=2, n_questions=0, seed=1)
    env.board1_landmarks = [(4, 4)]
    # unsorted on purpose
    env.board1_clues = [(3, 0), (1, 0)]

    # clue0 down
    env.sender_agent_action(2)

    assert env.board1_clues == [(3, 1), (1, 0)]

def test_receiver_action_preserves_guess_identity_order():
    env = BoardsImplementation(size=5, n_landmarks=2, n_clues=1, n_questions=0, seed=1)
    env.board2_questions = []
    env.board2_guesses = [(3, 0), (1, 0)]

    # guess0 DOWN
    env.receiver_agent_action(2)
    assert env.board2_guesses == [(3, 1), (1, 0)]
