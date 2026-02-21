import random
import numpy as np
import torch

from custom_types import Observation
from agent_architecture import PPOAgent, AgentParams


def make_const_obs(board_size: int, history_len: int, device: str = "cpu") -> Observation:
    # current_board: [1, 3, H, W]
    current = torch.zeros((1, 3, board_size, board_size), dtype=torch.float32, device=device)
    # previous_boards: [1, 3*history_len, H, W]
    prev = torch.zeros((1, 3 * history_len, board_size, board_size), dtype=torch.float32, device=device)
    # progress: [1, 1]
    prog = torch.zeros((1, 1), dtype=torch.float32, device=device)
    return Observation(current, prev, prog)


def greedy_action(agent: PPOAgent, obs: Observation) -> int:
    with torch.no_grad():
        dist = agent.actor(obs)
        # dist.logits shape: [B, n_actions], here B=1
        return int(torch.argmax(dist.logits, dim=-1).item())


def prob_of_action(agent: PPOAgent, obs: Observation, a: int) -> float:
    with torch.no_grad():
        dist = agent.actor(obs)
        p = torch.softmax(dist.logits, dim=-1)[0, a].item()
        return float(p)


def test_ppo_learns_trivial_bandit():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = "cpu"
    board_size = 4
    history_len = 1
    n_actions = 5
    target_action = 3

    params = AgentParams(
        gamma=0.0,
        gae_lambda=0.0,
        alpha=3e-3,
        policy_clip=0.2,
        batch_size=32,
        n_epochs=4,
        seed=0,
    )
    agent = PPOAgent(
        board_size=board_size,
        history_len=history_len,
        n_actions=n_actions,
        hidden_size=64,
        device=device,
        params=params,
    )

    obs = make_const_obs(board_size, history_len, device=device)

    p0 = prob_of_action(agent, obs, target_action)

    updates = 12
    episodes_per_update = 128

    for _ in range(updates):
        for _ in range(episodes_per_update):
            action, logp, val = agent.choose_action(obs)  # sampling, but seeded
            reward = 1.0 if action == target_action else 0.0
            done = True
            agent.remember(obs, action, logp, val, reward, done)

        agent.learn()

    p1 = prob_of_action(agent, obs, target_action)
    a_greedy = greedy_action(agent, obs)

    assert p1 > p0 + 0.20, (p0, p1)
    assert p1 > 0.60, (p0, p1)
    assert a_greedy == target_action