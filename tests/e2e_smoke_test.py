import pytest
import torch
import numpy as np

from env_wrapper import BoardsWrapper
from env_internals import BoardsImplementation
from agent_architecture import AgentParams, PPOAgent
from training_loop import train_agents


def create_env(*, size: int, n_landmarks: int, n_clues: int, n_questions: int, env_seed: int,
               max_moves: int, history_len: int, instant_reward_multiplier: float,
               end_reward_multiplier: float, device: str):
    env_internals = BoardsImplementation(size, n_landmarks, n_clues, n_questions, seed=env_seed)
    return BoardsWrapper(env_internals, max_moves, history_len, instant_reward_multiplier,
                         end_reward_multiplier, device)


def check_numbers(x):
    if x is None:
        return
    if isinstance(x, (float, int, np.floating, np.integer)):
        assert np.isfinite(x)
    elif isinstance(x, (list, tuple)):
        for v in x:
            check_numbers(v)
    elif isinstance(x, dict):
        for v in x.values():
            check_numbers(v)


@pytest.mark.e2e
def test_training_runs_smoke():
    """E2E smoke test: training runs end-to-end without exceptions."""
    device = "cpu"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)

    size = 4
    batch_size = 16
    n_epochs = 10
    training_epochs = 2
    n_episodes = 10
    hidden_size = size**2 * 32

    gamma = 0.99
    alpha = 2e-4
    gae_lambda = 0.95
    policy_clip = 0.1

    sender_seed = 135
    receiver_seed = 246

    env = create_env(size=size, n_landmarks=1, n_clues=1, n_questions=0, env_seed=12,
                     max_moves=size**2 * 4, history_len=4, instant_reward_multiplier=2.0,
                     end_reward_multiplier=10.0, device=device)

    sender_agent = PPOAgent(size, 4, env.sender_n_actions, hidden_size, device,
                            AgentParams(gamma=gamma, alpha=alpha, gae_lambda=gae_lambda,
                                        policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs,
                                        seed=sender_seed))
    receiver_agent = PPOAgent(size, 4, env.receiver_n_actions, hidden_size, device,
                              AgentParams(gamma=gamma, alpha=alpha, gae_lambda=gae_lambda,
                                          policy_clip=policy_clip, batch_size=batch_size,
                                          n_epochs=n_epochs, seed=receiver_seed))

    stats = None
    for _ in range(training_epochs):
        stats = train_agents(env, sender_agent, receiver_agent, n_episodes)

    assert stats is not None
    check_numbers(stats)
