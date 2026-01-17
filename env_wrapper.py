import torch
import numpy as np
from collections import deque
from env_internals import BoardsImplementation
from misc_utils import create_animation


class BoardsWrapper:
    def __init__(self, env: BoardsImplementation, max_moves: int, history_len: int, instant_multiplier: float, end_multiplier: float, device: str = "cpu"):
        self.env = env
        if max_moves < 1:
            raise ValueError("The number of moves in an episode should be positive.")
        if history_len < 1:
            raise ValueError("The size of visible history should be positive.")
        
        self.max_moves = max_moves
        self.history_len = history_len
        # Do nothing action, and 4 actions for every movable object on the board belonging to the agent.
        self.sender_n_actions = 1 + 4 * env.n_clues # Clues are moveable.
        self.receiver_n_actions = 1 + 4 * env.n_questions + 4 * env.n_landmarks # Question and guesses are moveable.
        # The number of guesses is equal to the number of landmarks.
        self.board_size = env.size # Both boards are square and have the same size.
        self.n_color_channels = 3
        self.store_n_states = max(history_len, 4)
        self._color_filter = np.array([[[1.0, 1.0, 0.0]] * self.board_size] * self.board_size)
        self.instant_multiplier = instant_multiplier
        self.end_multiplier = end_multiplier
        self.device = device
        self.reset()
        
    def reset(self):
        self.env.populate_boards()
        self.num_moves = 0
        self.done = False
        self.animation_frames = [self.env.draw_boards()]
        self.final_reward = None
        self.final_performance = None
        
        # Filling history of both agents with copies of the starting state and empty actions.
        self.sender_board_history = deque(self.store_n_states * [self.env.sender_agent_view()], self.store_n_states)
        self.sender_action_history = deque(self.store_n_states * [0], self.store_n_states)
        self.receiver_board_history = deque(self.store_n_states * [self.env.receiver_agent_view()], self.store_n_states)
        self.receiver_action_history = deque(self.store_n_states * [0], self.store_n_states)
    
    def _to_tensor(self, observation):
        current_board, previous_boards, progress = observation
        current_board = torch.FloatTensor(current_board).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        current_board = (current_board + 100) / 355
        previous_boards = torch.cat([torch.FloatTensor(board).unsqueeze(0).permute(0, 3, 1, 2).to(self.device) for board in previous_boards], dim = 1)
        previous_boards = (previous_boards + 100) / 355
        progress = torch.FloatTensor([progress]).unsqueeze(0).to(self.device)
        return current_board, previous_boards, progress
    
    def _end_episode(self):
        self.done = True
        self.final_reward, self.final_performance = self.env.reward_function()
        self.final_reward *= self.end_multiplier
        
    def _instant_reward(self, action_history, board_history):
        instant_reward = 0.0
        if False not in (board_history[-1] * self._color_filter == board_history[-3] * self._color_filter):
            instant_reward += -1.0 * self.instant_multiplier
        elif action_history[-1] == 0:
            instant_reward += -0.2 * self.instant_multiplier
        elif self.env.useless_action_flag:
            instant_reward += -0.5 * self.instant_multiplier
        else:
            instant_reward += 0.5 * self.instant_multiplier
        return instant_reward
    
    def render(self):
        title = f"Performance: {self.final_performance}" if self.done else None
        create_animation([self.animation_frames[0]] * 60 + self.animation_frames + [self.animation_frames[-1]] * 60, title)
        
    def sender_observe(self):
        current_board = self.env.sender_agent_view()
        previous_boards = list(self.sender_board_history)[-self.history_len:]
        progress = self.num_moves / self.max_moves
        return self._to_tensor((current_board, previous_boards, progress))
    
    def receiver_observe(self):
        current_board = self.env.receiver_agent_view()
        previous_boards = list(self.receiver_board_history)[-self.history_len:]
        progress = self.num_moves / self.max_moves
        return self._to_tensor((current_board, previous_boards, progress))
    
    def sender_act(self, action: int):
        if self.done:
            raise RuntimeError("The action limit was exhausted. Reset the environment.")
        self.num_moves += 1

        self.sender_board_history.append(self.env.sender_agent_view())
        self.sender_action_history.append(action)

        self.env.sender_agent_action(action)

        instant_reward = self._instant_reward(self.sender_action_history, self.sender_board_history)

        if self.num_moves >= self.max_moves:
            self._end_episode()
        
        self.animation_frames.append(self.env.draw_boards())
        return instant_reward, self.done
    
    def receiver_act(self, action: int):
        if self.done:
            raise RuntimeError("The action limit was exhausted. Reset the environment.")
        self.num_moves += 1

        self.receiver_board_history.append(self.env.receiver_agent_view())
        self.receiver_action_history.append(action)
        
        self.env.receiver_agent_action(action)
        
        instant_reward = self._instant_reward(self.receiver_action_history, self.receiver_board_history)

        if self.num_moves >= self.max_moves:
            self._end_episode()

        self.animation_frames.append(self.env.draw_boards())
        return instant_reward, self.done
    
    def get_useless_action_val(self):
        return int(self.env.useless_action_flag)
    
    def get_final_reward(self):
        if not self.done:
            raise RuntimeError("The episode is not over yet.")
        return self.final_reward
    
    def get_final_performance(self):
        if not self.done:
            raise RuntimeError("The episode is not over yet.")
        return self.final_performance
