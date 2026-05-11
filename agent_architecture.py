import math
import numpy as np
import pickle
from typing import cast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from scipy.optimize import linear_sum_assignment

import misc_utils
from custom_types import Observation

class PPOMemory:
    def __init__(self, batch_size: int, seed: int | None = None):
        if seed is None:
            seed = np.random.randint(42, 45954)
        elif seed < 0:
            raise ValueError("The seed should be non negative.")
        self.rng = np.random.default_rng(seed)
        self.batch_size: int = batch_size

        self.states: list[Observation] = []
        self.probs: list[float] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.vals: list[float] = []

        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        self.rng.shuffle(indices)
        batches = [indices[i: i + self.batch_size] for i in batch_start]
        
        states = self.states[:]
        actions = np.array(self.actions)
        probs = np.array(self.probs)
        vals = np.array(self.vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        return states, actions, probs, vals, rewards, dones, batches

    def store_memory(
            self,
            state: Observation | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            action: int,
            probs: float,
            vals: float,
            reward: float,
            done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class BoardGINEncoder(nn.Module):
    """GIN encoder over the board grid.

    Each cell is a node with features [channel_values..., x_norm, y_norm].
    Edges are 4-connected grid adjacency. Readout: project sum-pooled node
    embeddings at each layer (including input) and concatenate — the GIN
    readout from Xu et al. 2019, adapted from Abdelaziz et al. 2023.
    """

    def __init__(self, board_size: int, n_total_channels: int, hidden_dim: int, n_layers: int = 5):
        super().__init__()
        node_in_dim = n_total_channels + 2  # channel values + (x_norm, y_norm)
        self._board_size = board_size
        self._n_nodes = board_size * board_size
        self.out_dim = hidden_dim * (n_layers + 1)

        # GIN message-passing MLPs: h_new = MLP(h_self + sum_neighbors(h))
        self.gin_mlps = nn.ModuleList()
        for i in range(n_layers):
            in_d = node_in_dim if i == 0 else hidden_dim
            self.gin_mlps.append(nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ))

        # Readout: project each layer's sum-pooled output to hidden_dim then concat
        self.readout_projs = nn.ModuleList(
            [nn.Linear(node_in_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(n_layers + 1)]
        )

        # 4-connected grid edge index — fixed for lifetime of this module
        edges = []
        for y in range(board_size):
            for x in range(board_size):
                idx = y * board_size + x
                if x + 1 < board_size:
                    edges += [(idx, idx + 1), (idx + 1, idx)]
                if y + 1 < board_size:
                    edges += [(idx, idx + board_size), (idx + board_size, idx)]
        self.register_buffer('edge_index', torch.tensor(edges, dtype=torch.long).T.contiguous())

    def _node_features(self, boards: torch.Tensor) -> torch.Tensor:
        B, C, H, W = boards.shape
        channel_vals = boards.permute(0, 2, 3, 1).reshape(B, H * W, C)
        xs = torch.arange(W, dtype=boards.dtype, device=boards.device) / max(W - 1, 1)
        ys = torch.arange(H, dtype=boards.dtype, device=boards.device) / max(H - 1, 1)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([gx, gy], dim=-1).reshape(H * W, 2).unsqueeze(0).expand(B, -1, -1)
        return torch.cat([channel_vals, pos], dim=-1)  # [B, N, C+2]

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        B = boards.shape[0]
        N = self._n_nodes
        edge_index = cast(torch.Tensor, self.edge_index)
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h = self._node_features(boards)  # [B, N, node_in_dim]
        layer_pools = [self.readout_projs[0](h.sum(dim=1))]

        for k, mlp in enumerate(self.gin_mlps):
            d = h.shape[-1]
            agg = torch.zeros(B, N, d, device=h.device, dtype=h.dtype)
            agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, E, d), h[:, src, :])
            h = mlp(h + agg)
            layer_pools.append(self.readout_projs[k + 1](h.sum(dim=1)))

        return torch.cat(layer_pools, dim=-1)  # [B, out_dim]


class ActorCritic(nn.Module):
    def __init__(self, board_size: int, history_len: int, n_actions: int, hidden_size: int, n_channels_per_frame: int):
        super().__init__()
        n_total_channels = n_channels_per_frame * (history_len + 1)
        self.encoder = BoardGINEncoder(board_size, n_total_channels, hidden_dim=hidden_size)

        fc_in = self.encoder.out_dim + 1  # + progress scalar

        def mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(fc_in, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_dim),
            )

        self.actor_head = mlp(n_actions)
        self.critic_head = mlp(1)

    def forward(self, observation: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        boards = torch.cat([observation.previous_boards, observation.current_board], dim=1)
        z = self.encoder(boards)
        combined = torch.cat([z, observation.progress], dim=1)
        logits = self.actor_head(combined)
        value = self.critic_head(combined).squeeze(-1)
        return logits, value

    def dist_and_value(self, observation: Observation) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.forward(observation)
        return Categorical(logits=logits), value
    
class AgentParams:
    def __init__(self, gamma = 0.99, alpha = 1e-4, gae_lambda = 0.95, policy_clip = 0.1, batch_size = 8, n_epochs = 4, seed = None, entropy_coeff = 0.01):
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.entropy_coeff = entropy_coeff

class PPOAgent:
    def __init__(self, board_size, history_len, n_actions, hidden_size, device: str = "cpu", params: AgentParams | None = None, frozen: bool = False, n_channels_per_frame: int = 3):
        if params is None:
            self.params = AgentParams()
        else:
            self.params = params

        self.board_size = board_size
        self.history_len = history_len
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.device = device
        self.frozen = frozen
        self.n_channels_per_frame = n_channels_per_frame

        ac = ActorCritic(board_size, history_len, n_actions, hidden_size, n_channels_per_frame).to(device)
        try:
            ac = cast(ActorCritic, torch.compile(ac, mode="reduce-overhead", dynamic=True))
        except Exception:
            pass
        self.ac = ac
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.params.alpha)
        self.memory = PPOMemory(self.params.batch_size, self.params.seed)
        
    def freeze(self, frozen: bool):
        self.frozen = frozen
       
    def remember(self, state, action, probs, vals, reward, done):
        if self.frozen:
            return
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def policy(self, observation: Observation) -> Categorical:
        dist, _ = self.ac.dist_and_value(observation)
        return dist

    def value(self, observation: Observation) -> torch.Tensor:
        _, v = self.ac.dist_and_value(observation)
        return v

    def choose_action(self, observation: Observation) -> tuple[int, float, float]:
        with torch.no_grad():
            dist, value = self.ac.dist_and_value(observation)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def batch_choose_action(self, observation: Observation) -> tuple[list[int], list[float], list[float]]:
        with torch.no_grad():
            dist, values = self.ac.dist_and_value(observation)
            actions = dist.sample()
            logps = dist.log_prob(actions)
        return actions.tolist(), logps.tolist(), values.tolist()

    def learn(self):
        if self.frozen:
            return

        entropy_dist, actor_loss_dist, critic_loss_dist, total_loss_dist = [], [], [], []

        for _ in range(self.params.n_epochs):
            state_list, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            t_steps = len(action_arr)
            if t_steps == 0:
                self.memory.clear_memory()
                return entropy_dist, actor_loss_dist, critic_loss_dist, total_loss_dist

            last_value = 0.0 if bool(dones_arr[-1]) else float(vals_arr[-1])

            advantage_np = misc_utils.compute_gae(
                reward_arr,
                vals_arr,
                dones_arr,
                gamma=self.params.gamma,
                gae_lambda=self.params.gae_lambda,
                last_value=last_value,
            )
            advantage_np = (advantage_np - advantage_np.mean()) / (advantage_np.std() + 1e-8)
            advantage = torch.from_numpy(advantage_np).to(self.device)
            values = torch.tensor(vals_arr, dtype=torch.float32, device=self.device)

            for batch in batches:
                current_boards = torch.cat([state_list[i].current_board for i in batch], dim=0).to(self.device)
                previous_boards = torch.cat([state_list[i].previous_boards for i in batch], dim=0).to(self.device)
                progresses = torch.cat([state_list[i].progress for i in batch], dim=0).to(self.device)
                states = Observation(current_boards, previous_boards, progresses)

                old_log_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32, device=self.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.long, device=self.device)

                dist, critic_value = self.ac.dist_and_value(states)
                critic_value = torch.squeeze(critic_value)

                new_log_probs = dist.log_prob(actions)
                prob_ratio = (new_log_probs - old_log_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.params.policy_clip, 1 + self.params.policy_clip
                ) * advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value).pow(2).mean()

                entropy = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - self.params.entropy_coeff * entropy

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                entropy_dist.append(float(entropy.item()))
                actor_loss_dist.append(float(actor_loss.item()))
                critic_loss_dist.append(float(critic_loss.item()))
                total_loss_dist.append(float(total_loss.item()))

        self.memory.clear_memory()
        return entropy_dist, actor_loss_dist, critic_loss_dist, total_loss_dist
        
class GreedySenderAgent:
    """Scripted sender that greedily moves the farthest clue one step toward its landmark.

    Locks the clue→landmark assignment once per episode (when env.num_moves == 0) so
    gradients received by a co-trained receiver remain consistent within the episode.
    Requires the env to expose .num_moves, .env.board1_clues, and .env.board1_landmarks.
    """

    def __init__(self, env) -> None:
        self._impl = env.env
        self._wrapper = env
        self._locked_assignment: list[tuple[int, int]] | None = None

    def choose_action(self, _obs=None) -> tuple[int, float, float]:
        if self._wrapper.num_moves == 0:
            clues = self._impl.board1_clues
            landmarks = self._impl.board1_landmarks
            cost = np.array(
                [[math.sqrt((c[0] - l[0]) ** 2 + (c[1] - l[1]) ** 2) for l in landmarks] for c in clues],
                dtype=float,
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            self._locked_assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

        best_action, best_dist = 0, 0.0
        for ci, li in self._locked_assignment:  # type: ignore[union-attr]
            cx, cy = self._impl.board1_clues[ci]
            lx, ly = self._impl.board1_landmarks[li]
            dist = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            if dist <= best_dist:
                continue
            best_dist = dist
            dx, dy = lx - cx, ly - cy
            if abs(dx) >= abs(dy) and dx != 0:
                direction = 3 if dx > 0 else 2
            elif dy != 0:
                direction = 1 if dy > 0 else 0
            else:
                continue
            best_action = 1 + 4 * ci + direction
        return best_action, 0.0, 0.0


class RandomAgent:
    def __init__(self, permitted_actions: list[int], seed: int | None = None):
        if seed is None:
            seed = np.random.randint(42, 45954)
        elif seed < 0:
            raise ValueError("The seed should be non negative.")
        self.rng = np.random.default_rng(seed)
        self.permitted_actions = permitted_actions

    def choose_action(self, _):
        return int(self.rng.choice(self.permitted_actions)), 0.0, 0.0

    def batch_choose_action(self, _: Observation) -> tuple[list[int], list[float], list[float]]:
        raise NotImplementedError("RandomAgent does not support batch_choose_action")

def save_agents(sender: PPOAgent | RandomAgent, receiver: PPOAgent | RandomAgent, file_path: str):
    checkpoint = {"sender": sender, "receiver": receiver}
    with open(file_path, 'wb') as file:
        pickle.dump(checkpoint, file)

def load_agents(file_path: str):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data["sender"], loaded_data["receiver"]