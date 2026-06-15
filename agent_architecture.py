import math

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.distributions import Categorical

import misc_utils
from custom_types import Observation


def _compute_max_nodes(board_size: int) -> int:
    """Returns total nodes in a full quadtree for a `board_size` * `board_size` grid."""
    max_depth = int(math.log2(board_size))
    return (4 ** (max_depth + 1) - 1) // 3


def _compute_bfs_regions(board_size: int) -> list[tuple[int, int, int, int]]:
    """Returns (y1, y2, x1, x2) for each node in BFS order."""
    regions = [(0, board_size, 0, board_size)]
    frontier = regions[:]
    while frontier[0][1] - frontier[0][0] > 1:
        next_frontier: list[tuple[int, int, int, int]] = []
        for (y1, y2, x1, x2) in frontier:
            mh, mw = (y1 + y2) // 2, (x1 + x2) // 2
            children = [(y1, mh, x1, mw), (y1, mh, mw, x2), (mh, y2, x1, mw), (mh, y2, mw, x2)]
            regions.extend(children)
            next_frontier.extend(children)
        frontier = next_frontier
    return regions


def _compute_adjacency_matrix(max_nodes: int) -> torch.Tensor:
    """Returns fixed undirected adjacency for a full quadtree in BFS order."""
    A = torch.zeros(max_nodes, max_nodes)
    for i in range(1, max_nodes):
        p = (i - 1) // 4
        A[i, p] = A[p, i] = 1.0
    return A


class GINLayer(nn.Module):
    """Single Graph Isomorphism Network message-passing layer."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: [B, N, in_dim], adj: [B, N, N]
        return self.mlp(h + torch.bmm(adj, h))


class GINEncoder(nn.Module):
    """
    Implements the architecture from Abdel-Aziz et al. (arXiv:2306.11336):
      1. Build full-resolution quadtree (fixed structure for given board_size).
      2. GIN on full tree -> graph representation hG.
      3. MLP(hG, progress) -> ST-Gumbel-Softmax -> keep/merge per internal node.
      4. Propagate decisions -> active node mask -> abstracted graph.
      5. GIN on abstracted tree -> sum-pooled output.

    Note that it requires `board_size` to be a power of 2.
    """
    def __init__(
        self,
        board_size: int,
        history_len: int,
        n_channels_per_frame: int,
        gin_hidden: int = 64,
        gin_layers: int = 3,
        gumbel_temperature: float = 1.0,
    ):
        super().__init__()
        if board_size & (board_size - 1):
            raise ValueError(f"GINEncoder requires board_size to be a power of 2, got {board_size}")

        self.board_size = board_size
        self.n_ch = n_channels_per_frame * (history_len + 1)
        self.max_depth = int(math.log2(board_size))
        self.max_nodes = _compute_max_nodes(board_size)
        # Internal nodes = all except the leaf level (board_size^2 leaves)
        self.internal_count = (board_size ** 2 - 1) // 3
        # Per-channel means + depth + cy + cx + region_size
        self.feat_dim = self.n_ch + 4
        self.gumbel_temperature = gumbel_temperature

        self.regions = _compute_bfs_regions(board_size)
        pos = torch.zeros(self.max_nodes, 4)
        for i, (y1, y2, x1, x2) in enumerate(self.regions):
            depth = int(math.log2(board_size / (y2 - y1)))
            pos[i, 0] = depth / self.max_depth
            pos[i, 1] = (y1 + y2) / 2 / board_size
            pos[i, 2] = (x1 + x2) / 2 / board_size
            pos[i, 3] = (y2 - y1) * (x2 - x1) / (board_size ** 2)
        # [max_nodes, 4]
        self.register_buffer('pos_feats', pos)
        # [max_nodes, max_nodes]
        self.register_buffer('A', _compute_adjacency_matrix(self.max_nodes))
        # Projected input + K layer outputs
        hG_dim = gin_hidden * (gin_layers + 1)

        # GIN on full tree
        self.feat_proj1 = nn.Linear(self.feat_dim, gin_hidden)
        self.gin1 = nn.ModuleList([
            GINLayer(self.feat_dim if i == 0 else gin_hidden, gin_hidden)
            for i in range(gin_layers)
        ])

        # Abstractor: hG + progress scalar -> keep/merge logits for internal nodes
        self.abstractor = nn.Sequential(
            nn.Linear(hG_dim + 1, gin_hidden * 2),
            nn.ReLU(),
            nn.Linear(gin_hidden * 2, self.internal_count * 2),
        )

        # GIN on abstracted tree
        self.feat_proj2 = nn.Linear(self.feat_dim, gin_hidden)
        self.gin2 = nn.ModuleList([
            GINLayer(self.feat_dim if i == 0 else gin_hidden, gin_hidden)
            for i in range(gin_layers)
        ])

        # Same shape as `gin1` readout
        self.out_dim = hG_dim

    def _node_features(self, boards: torch.Tensor) -> torch.Tensor:
        """Returns node features for each board in the batch."""
        B, C = boards.shape[0], boards.shape[1]
        X = torch.zeros(B, self.max_nodes, self.feat_dim, device=boards.device)
        for i, (y1, y2, x1, x2) in enumerate(self.regions):
            X[:, i, :C] = boards[:, :, y1:y2, x1:x2].mean(dim=[2, 3])
        X[:, :, C:] = self.pos_feats.unsqueeze(0)
        return X

    def _gin_readout(
        self,
        gin_layers: nn.ModuleList,
        feat_proj: nn.Linear,
        X: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Sum-pool each GIN layer output + projected input, concatenate -> [B, hG_dim]."""
        pools = [feat_proj(X).sum(dim=1)]
        h = X
        for layer in gin_layers:
            h = layer(h, A)
            pools.append(h.sum(dim=1))
        return torch.cat(pools, dim=-1)

    def _propagate_keep(self, keep: torch.Tensor) -> torch.Tensor:
        """
        Propagates keep decisions [B, internal_count]: 1=keep node as internal,
        0=merge (discard children). Returns active [B, max_nodes]: 1 if node survives
        into the abstracted tree. Node i is active iff every ancestor decided to keep
        (not merge) it.
        """
        # Build as a list to avoid in-place ops on autograd tensors
        ones = torch.ones(keep.shape[0], device=keep.device)
        active: list[torch.Tensor] = [ones]  # root always active
        for i in range(1, self.max_nodes):
            p = (i - 1) // 4
            # Child is active if its parent is active and parent didn't merge
            active.append(active[p] * keep[:, p])
        return torch.stack(active, dim=1)  # [B, max_nodes]

    def forward(self, observation: Observation) -> torch.Tensor:
        boards = torch.cat([observation.previous_boards, observation.current_board], dim=1)
        B = boards.shape[0]
        X = self._node_features(boards)
        A = self.A.unsqueeze(0).expand(B, -1, -1)

        # GIN on full tree
        hG = self._gin_readout(self.gin1, self.feat_proj1, X, A)

        # Getting learn merge decisions
        logits = self.abstractor(torch.cat([hG, observation.progress], dim=-1))
        # [B, internal_count]: 1=keep, 0=merge
        keep = F.gumbel_softmax(
            logits.view(B, self.internal_count, 2),
            tau=self.gumbel_temperature, hard=True,
        )[..., 0]

        # [B, max_nodes]
        active = self._propagate_keep(keep)

        X_abs = X * active.unsqueeze(-1)
        A_abs = self.A.unsqueeze(0) * (active.unsqueeze(2) * active.unsqueeze(1))

        # GIN on abstracted tree
        return self._gin_readout(self.gin2, self.feat_proj2, X_abs, A_abs)

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

class SharedEncoder(nn.Module):
    def __init__(
        self,
        board_size: int,
        history_len: int,
        n_channels_per_frame: int,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.channels = n_channels_per_frame * (history_len + 1)
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.channels, self.embedding_dim),
            nn.ReLU(),
        )
        self.out_dim = self.embedding_dim

    def forward(self, observation):
        x = torch.cat([observation.previous_boards, observation.current_board], dim=1)
        return self.conv(x)

class ActorCritic(nn.Module):
    def __init__(
        self,
        board_size: int,
        history_len: int,
        n_actions: int,
        hidden_size: int,
        n_channels_per_frame: int,
        encoder: str = "cnn",
        gin_hidden: int = 64,
        gin_layers: int = 3,
        cnn_embedding_dim: int = 128,
    ):
        super().__init__()
        if encoder == "gin":
            self.encoder = GINEncoder(board_size, history_len, n_channels_per_frame, gin_hidden, gin_layers)
        else:
            self.encoder = SharedEncoder(board_size, history_len, n_channels_per_frame, cnn_embedding_dim)

        fc_in = self.encoder.out_dim + 1  # + progress scalar

        def mlp(out_dim: int):
            return nn.Sequential(
                nn.Linear(fc_in, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_dim),
            )

        self.actor_head = mlp(n_actions)
        self.critic_head = mlp(1)

    def forward(self, observation):
        z = self.encoder(observation)
        combined = torch.cat([z, observation.progress], dim=1)
        logits = self.actor_head(combined)

        value = self.critic_head(torch.cat([z, observation.progress], dim=1)).squeeze(-1)

        return logits, value

    def dist_and_value(self, observation):
        logits, value = self.forward(observation)
        dist = Categorical(logits=logits)
        return dist, value
    
class AgentParams:
    def __init__(self, gamma = 0.99, alpha = 1e-4, gae_lambda = 0.95, policy_clip = 0.1, batch_size = 8, n_epochs = 4, seed = None):
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed

class PPOAgent:
    def __init__(
        self,
        board_size,
        history_len,
        n_actions,
        hidden_size,
        device: str = "cpu",
        params: AgentParams | None = None,
        frozen: bool = False,
        n_channels_per_frame: int = 3,
        encoder: str = "cnn",
        gin_hidden: int = 64,
        gin_layers: int = 3,
        cnn_embedding_dim: int = 128,
    ):
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
        self.encoder_type = encoder
        self.gin_hidden = gin_hidden
        self.gin_layers = gin_layers
        self.cnn_embedding_dim = cnn_embedding_dim

        self.ac = ActorCritic(
            board_size, history_len, n_actions, hidden_size, n_channels_per_frame,
            encoder=encoder, gin_hidden=gin_hidden, gin_layers=gin_layers,
            cnn_embedding_dim=cnn_embedding_dim,
        ).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.params.alpha)
        self.memory = PPOMemory(self.params.batch_size, self.params.seed)
        
    def freeze(self, frozen: bool):
        self.frozen = frozen

    @staticmethod
    def _copy_matching_state_dict(source: nn.Module, target: nn.Module) -> None:
        source_state = source.state_dict()
        target_state = target.state_dict()
        matched = {
            key: value
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        target.load_state_dict(matched, strict=False)

    def adapt_to_stage(
        self,
        new_board_size: int,
        new_n_actions: int,
        new_encoder_type: str,
        new_hidden_size: int,
        new_n_channels_per_frame: int,
    ) -> bool:
        if self.encoder_type != new_encoder_type:
            return False
        if self.hidden_size != new_hidden_size:
            return False
        if self.n_channels_per_frame != new_n_channels_per_frame:
            return False
        if new_board_size == self.board_size and new_n_actions == self.n_actions:
            return True

        old_ac = self.ac
        new_ac = ActorCritic(
            board_size=new_board_size,
            history_len=self.history_len,
            n_actions=new_n_actions,
            hidden_size=self.hidden_size,
            n_channels_per_frame=self.n_channels_per_frame,
            encoder=self.encoder_type,
            gin_hidden=self.gin_hidden,
            gin_layers=self.gin_layers,
            cnn_embedding_dim=self.cnn_embedding_dim,
        ).to(self.device)

        try:
            self._copy_matching_state_dict(old_ac.encoder, new_ac.encoder)
        except RuntimeError:
            return False

        with torch.no_grad():
            self._copy_matching_state_dict(old_ac.actor_head, new_ac.actor_head)
            new_ac.critic_head.load_state_dict(old_ac.critic_head.state_dict())

        self.ac = new_ac
        self.board_size = new_board_size
        self.n_actions = new_n_actions
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.params.alpha)
        return True

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
                total_loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy

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
        
class RandomAgent:
    def __init__(self, permitted_actions: list[int], seed: int | None = None):
        if seed is None:
            seed = np.random.randint(42, 45954)
        elif seed < 0:
            raise ValueError("The seed should be non negative.")
        self.rng = np.random.default_rng(seed)
        self.permitted_actions = permitted_actions
    
    def choose_action(self, _) -> tuple[int, float, float]:
        return int(self.rng.choice(self.permitted_actions)), 0.0, 0.0

class CopycatReceiverAgent:
    """Receiver that mirrors clue shadow positions onto guesses every turn.

    Rather than moving step-by-step, it directly writes board2_guesses to the
    current clue positions each turn, then returns action 0 so receiver_act
    does nothing further. This gives an instantaneous, perfect copy of where
    the sender placed its clues, making the final reward a clean proxy for
    sender navigation quality with no step-count or blocking artifacts.

    Assignment (which guess tracks which clue) is locked via Hungarian matching
    at the start of each episode and held fixed for the full episode.
    """

    def __init__(self, env) -> None:
        self.env = env
        self._assignment: list[tuple[int, int]] = []

    def _lock_assignment(self) -> None:
        guesses = self.env.env.board2_guesses
        shadows = self.env.env.board1_clues
        cost = np.array([
            [math.sqrt((guesses[g][0] - shadows[s][0]) ** 2 + (guesses[g][1] - shadows[s][1]) ** 2)
             for s in range(len(shadows))]
            for g in range(len(guesses))
        ])
        row_ind, col_ind = linear_sum_assignment(cost)
        self._assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

    def choose_action(self, _) -> tuple[int, float, float]:
        if self.env.num_moves == 1:
            self._lock_assignment()

        shadows = self.env.env.board1_clues
        for guess_idx, shadow_idx in self._assignment:
            self.env.env.board2_guesses[guess_idx] = shadows[shadow_idx]

        return 0, 0.0, 0.0


def save_agents(sender: PPOAgent | RandomAgent, receiver: PPOAgent | RandomAgent, file_path: str):
    checkpoint = {"sender": sender, "receiver": receiver}
    with open(file_path, 'wb') as file:
        pickle.dump(checkpoint, file)

def load_agents(file_path: str):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data["sender"], loaded_data["receiver"]