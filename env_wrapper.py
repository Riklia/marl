import torch
import numpy as np
from collections import deque
from collections.abc import Sequence
from typing import Final


from env_internals import BoardsImplementation
from misc_utils import create_animation
from custom_types import Observation


class BoardsWrapper:
    def __init__(
            self,
            env: BoardsImplementation,
            max_moves: int,
            history_len: int,
            instant_multiplier: float,
            end_multiplier: float,
            device: str = "cpu"
    ) -> None:
        self.env = env
        if max_moves < 1:
            raise ValueError("The number of moves in an episode should be positive.")
        if history_len < 1:
            raise ValueError("The size of visible history should be positive.")

        self.max_moves: int = max_moves
        self.history_len: int = history_len

        self.sender_n_actions: int = 1 + 4 * env.n_clues
        self.receiver_n_actions: int = 1 + 4 * env.n_questions + 4 * env.n_landmarks

        self.board_size: int = env.size
        self.n_color_channels: Final[int] = 3

        self.store_n_states: int = max(history_len, 4)

        # shape: [H,W,3] filter for numpy operations
        self._color_filter: np.ndarray = np.array([[[1.0, 1.0, 0.0]] * self.board_size] * self.board_size)

        self.instant_multiplier: float = instant_multiplier
        self.end_multiplier: float = end_multiplier
        self.device: str = device

        self.num_moves: int = 0
        self.done: bool = False
        self.animation_frames: list[np.ndarray] = []
        self.final_reward: float | None = None
        self.final_performance: float | None = None

        # History containers are initialized in `reset`.
        self.sender_board_history: deque[np.ndarray]
        self.sender_action_history: deque[int]
        self.receiver_board_history: deque[np.ndarray]
        self.receiver_action_history: deque[int]

        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self) -> None:
        self.env.populate_boards()
        self.num_moves = 0
        self.done = False
        self.animation_frames = [self.env.draw_boards()]
        self.final_reward = None
        self.final_performance = None

        self.sender_board_history = deque(
            [self.env.sender_agent_view()] * self.store_n_states,
            maxlen=self.store_n_states,
        )
        self.sender_action_history = deque([0] * self.store_n_states, maxlen=self.store_n_states)

        self.receiver_board_history = deque(
            [self.env.receiver_agent_view()] * self.store_n_states,
            maxlen=self.store_n_states,
        )
        self.receiver_action_history = deque([0] * self.store_n_states, maxlen=self.store_n_states)

    def _to_tensor(
        self,
        current_board: np.ndarray,
        previous_boards: list[np.ndarray],
        progress: float,
    ) -> Observation:
        # current_board: H,W,3  -> [1,3,H,W]
        cur = torch.as_tensor(current_board, dtype=torch.float32, device=self.device)
        cur = cur.unsqueeze(0).permute(0, 3, 1, 2)
        cur = (cur + 100) / 355

        # previous_boards: list of H,W,3 -> [1, 3*len, H, W]
        prev_list = [
            torch.as_tensor(b, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
            for b in previous_boards
        ]
        prev = torch.cat(prev_list, dim=1)
        prev = (prev + 100) / 355

        prog = torch.tensor([[progress]], dtype=torch.float32, device=self.device)

        return Observation(current_board=cur, previous_boards=prev, progress=prog)
    
    def _end_episode(self) -> None:
        self.done = True
        final_reward, final_perf = self.env.reward_function()
        self.final_reward = float(final_reward) * self.end_multiplier
        self.final_performance = float(final_perf)

    def _instant_reward(self, action_history: deque[int], board_history: deque[np.ndarray]) -> float:
        if False not in (board_history[-1] * self._color_filter == board_history[-3] * self._color_filter):
            return -1.0 * self.instant_multiplier
        if action_history[-1] == 0:
            return -0.2 * self.instant_multiplier
        if self.env.useless_action_flag:
            return -0.5 * self.instant_multiplier
        return 0.5 * self.instant_multiplier
    
    def render(self) -> None:
        title = f"Performance: {self.final_performance}" if self.done else None
        create_animation(
            [self.animation_frames[0]] * 60 + self.animation_frames + [self.animation_frames[-1]] * 60,
            title,
        )

    def sender_observe(self) -> Observation:
        current_board = self.env.sender_agent_view()
        previous_boards = list(self.sender_board_history)[-self.history_len :]
        progress = self.num_moves / self.max_moves
        return self._to_tensor(current_board, previous_boards, progress)

    def receiver_observe(self) -> Observation:
        current_board = self.env.receiver_agent_view()
        previous_boards = list(self.receiver_board_history)[-self.history_len :]
        progress = self.num_moves / self.max_moves
        return self._to_tensor(current_board, previous_boards, progress)
    
    def sender_act(self, action: int) -> tuple[float, bool]:
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
    
    def receiver_act(self, action: int) -> tuple[float, bool]:
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
    
    def get_useless_action_val(self) -> int:
        return int(self.env.useless_action_flag)

    def get_final_reward(self) -> float:
        if not self.done or self.final_reward is None:
            raise RuntimeError("The episode is not over yet.")
        return self.final_reward

    def get_final_performance(self) -> float:
        if not self.done or self.final_performance is None:
            raise RuntimeError("The episode is not over yet.")
        return self.final_performance
