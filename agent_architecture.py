import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOMemory:
    def __init__(self, batch_size: int, seed: int | None = None):
        if seed is None:
            seed = np.random.randint(42, 45954)
        elif seed < 0:
            raise ValueError("The seed should be non negative.")
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.states) - 1 # Not using the dummy memory for training.
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

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = list()
        self.probs = list()
        self.actions = list()
        self.rewards = list()
        self.dones = list()
        self.vals = list()

class ActorNetwork(nn.Module):
    def __init__(self, board_size, history_len, n_actions, hidden_size, alpha):
        super(ActorNetwork, self).__init__()
        
        channels = 3 * (history_len + 1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        fc_in_size = channels * board_size * board_size + 1
        self.fc = nn.Sequential(
            nn.Linear(fc_in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim = -1)
        )

        self.optimizer = optim.SGD(self.parameters(), lr = alpha) # type: ignore

    def forward(self, observation):
        current_board, previous_boards, progress = observation
        combined = torch.cat([previous_boards.flatten(1), current_board.flatten(1), progress], dim = 1)
        return Categorical(self.fc(combined))

class CriticNetwork(nn.Module):
    def __init__(self, board_size, history_len, hidden_size, alpha):
        super(CriticNetwork, self).__init__()
        
        channels = 3 * (history_len + 1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        fc_in_size = channels * board_size * board_size + 1
        self.fc = nn.Sequential(
            nn.Linear(fc_in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.SGD(self.parameters(), lr = alpha) # type: ignore

    def forward(self, observation):
        current_board, previous_boards, progress = observation
        combined = torch.cat([previous_boards.flatten(1), current_board.flatten(1), progress], dim = 1)
        return self.fc(combined)
    
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
    def __init__(self, board_size, history_len, n_actions, hidden_size, device: str = "cpu", params: AgentParams | None = None, frozen: bool = False):
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
        
        self.actor = ActorNetwork(board_size, history_len, n_actions, hidden_size, self.params.alpha).to(device)
        self.critic = CriticNetwork(board_size, history_len, hidden_size, self.params.alpha).to(device)
        self.memory = PPOMemory(self.params.batch_size, self.params.seed)
        
    def freeze(self, frozen: bool):
        self.frozen = frozen
       
    def remember(self, state, action, probs, vals, reward, done):
        if self.frozen:
            return
        self.memory.store_memory(state, action, probs, vals, reward, done)
        if done: # Dummy memory at the end because otherwise the end reward is ignored.
            self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        with torch.no_grad():
            dist = self.actor(observation)
            value = self.critic(observation)
            action = dist.sample()

            probs = torch.squeeze(dist.log_prob(action)).item()
            action = int(torch.squeeze(action).item())
            value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        if self.frozen:
            return
        for _ in range(self.params.n_epochs):
            state_list, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype = np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.params.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.params.gamma * self.params.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            for batch in batches:
                current_boards = torch.cat([state_list[i][0] for i in batch], dim = 0).to(self.device)
                previous_boards = torch.cat([state_list[i][1] for i in batch], dim = 0).to(self.device)
                progresses = torch.cat([state_list[i][2] for i in batch], dim = 0).to(self.device)
                states = (current_boards, previous_boards, progresses)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.params.policy_clip, 1 + self.params.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                entropy = dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        
class RandomAgent:
    def __init__(self, permitted_actions: list[int], seed: int | None = None):
        if seed is None:
            seed = np.random.randint(42, 45954)
        elif seed < 0:
            raise ValueError("The seed should be non negative.")
        self.rng = np.random.default_rng(seed)
        self.permitted_actions = permitted_actions
    
    def choose_action(self, _): # Ignore observation.
        return self.rng.choice(self.permitted_actions)

def save_agents(sender: PPOAgent | RandomAgent, receiver: PPOAgent | RandomAgent, file_path: str):
    checkpoint = {"sender": sender, "receiver": receiver}
    with open(file_path, 'wb') as file:
        pickle.dump(checkpoint, file)

def load_agents(file_path: str):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data["sender"], loaded_data["receiver"]