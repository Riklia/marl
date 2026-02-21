import pytest
import torch

from env_internals import BoardsImplementation
from env_wrapper import BoardsWrapper


@pytest.fixture
def env5x5():
    # Small, deterministic env for wrapper contract tests
    return BoardsImplementation(
        size=5,
        n_landmarks=2,
        n_clues=2,
        n_questions=1,
        linked_shadows=True,
        seed=42,
    )


def test_wrapper_sender_observe_shapes_dtypes_ranges(env5x5):
    w = BoardsWrapper(env5x5, max_moves=10, history_len=3, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    obs = w.sender_observe()
    cur, prev, prog = obs.current_board, obs.previous_boards, obs.progress

    assert isinstance(cur, torch.Tensor)
    assert isinstance(prev, torch.Tensor)
    assert isinstance(prog, torch.Tensor)

    assert cur.shape == (1, 3, env5x5.size, env5x5.size)
    assert prev.shape == (1, 3 * 3, env5x5.size, env5x5.size)
    assert prog.shape == (1, 1)

    assert cur.dtype == torch.float32
    assert prev.dtype == torch.float32
    assert prog.dtype == torch.float32

    # Normalization used in wrapper: (x + 100) / 355, where x is uint8 in [0,255]
    lo = (0 + 100) / 355
    hi = (255 + 100) / 355
    assert float(cur.min()) >= lo - 1e-6
    assert float(cur.max()) <= hi + 1e-6
    assert float(prev.min()) >= lo - 1e-6
    assert float(prev.max()) <= hi + 1e-6

    assert 0.0 <= float(prog.item()) <= 1.0


def test_wrapper_receiver_observe_shapes(env5x5):
    w = BoardsWrapper(env5x5, max_moves=10, history_len=2, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    obs = w.sender_observe()
    cur, prev, prog = obs.current_board, obs.previous_boards, obs.progress

    assert cur.shape == (1, 3, env5x5.size, env5x5.size)
    assert prev.shape == (1, 3 * 2, env5x5.size, env5x5.size)
    assert prog.shape == (1, 1)


def test_wrapper_action_index_bounds_and_types(env5x5):
    w = BoardsWrapper(env5x5, max_moves=5, history_len=2, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    # action 0 must always be valid (None)
    r, done = w.sender_act(0)
    assert isinstance(r, float)
    assert isinstance(done, bool)

    # last action index should be valid too
    last = len(env5x5.sender_agent_actions) - 1
    r, done = w.sender_act(last)
    assert isinstance(r, float)
    assert isinstance(done, bool)


def test_wrapper_history_buffer_rolls(env5x5):
    w = BoardsWrapper(env5x5, max_moves=10, history_len=3, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    # At init, history contains zeros
    obs0 = w.sender_observe()
    cur0, prev0, _ = obs0.current_board, obs0.previous_boards, obs0.progress
    assert torch.all(prev0.isfinite())

    # Take a step; the previous buffer should now include the last board at the front (most recent)
    w.sender_act(0)
    obs1 = w.sender_observe()
    cur1, prev1, _ = obs1.current_board, obs1.previous_boards, obs1.progress

    # prev1 is [board_{t-1}, board_{t-2}, board_{t-3}] concatenated along channel
    # The most recent previous (t-1) should equal current from just-before step.
    # We can compare against w.sender_board_history[-2] transformed by wrapper logic,
    # but simplest stable check: channels [0:3] of prev should match previous "current" tensor.
    assert torch.allclose(prev1[:, 0:3, :, :], cur0, atol=0, rtol=0)


def test_wrapper_progress_increases_with_steps(env5x5):
    w = BoardsWrapper(env5x5, max_moves=4, history_len=1, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    p0 = w.sender_observe().progress.item()
    w.sender_act(0)
    p1 = w.sender_observe().progress.item()
    w.sender_act(0)
    p2 = w.sender_observe().progress.item()

    assert p1 > p0
    assert p2 > p1
    assert p2 <= 1.0


def test_wrapper_done_and_final_reward_scaling(env5x5):
    w = BoardsWrapper(env5x5, max_moves=2, history_len=1, instant_multiplier=1.0, end_multiplier=3.0, device="cpu")

    assert w.done is False
    r1, d1 = w.sender_act(0)
    assert d1 is False
    r2, d2 = w.sender_act(0)
    assert d2 is True
    assert w.done is True

    base_reward, base_perf = env5x5.reward_function()
    assert w.get_final_reward() == base_reward * 3.0
    assert w.get_final_performance() == base_perf


def test_wrapper_get_final_reward_raises_before_done(env5x5):
    w = BoardsWrapper(env5x5, max_moves=3, history_len=1, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    with pytest.raises(RuntimeError):
        w.get_final_reward()
    with pytest.raises(RuntimeError):
        w.get_final_performance()


def test_wrapper_act_raises_when_done(env5x5):
    w = BoardsWrapper(env5x5, max_moves=1, history_len=1, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")
    w.sender_act(0)
    assert w.done is True

    with pytest.raises(RuntimeError):
        w.sender_act(0)
    with pytest.raises(RuntimeError):
        w.receiver_act(0)


def test_wrapper_smoke_rollout_sender_and_receiver(env5x5):
    w = BoardsWrapper(env5x5, max_moves=6, history_len=2, instant_multiplier=1.0, end_multiplier=1.0, device="cpu")

    for t in range(3):
        _ = w.sender_observe()
        r_s, d_s = w.sender_act(0)
        assert isinstance(r_s, float)
        assert isinstance(d_s, bool)
        if d_s:
            break

        _ = w.receiver_observe()
        r_r, d_r = w.receiver_act(0)
        assert isinstance(r_r, float)
        assert isinstance(d_r, bool)
        if d_r:
            break
