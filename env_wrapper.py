import torch
import numpy as np
from collections import deque


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
            device: str = "cpu",
            perf_epsilon: float = 1e-6,
            shaping_multiplier: float = 0.0,
            sender_shaping_multiplier: float = 0.0,
            gamma: float = 0.99,
            shaping_gamma: float | None = None,
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
        self.sender_n_channels: int = env.n_sender_channels
        self.receiver_n_channels: int = env.n_receiver_channels

        self.board_size: int = env.size

        self.store_n_states: int = max(history_len, 4)

        # Per-agent filters: ones everywhere except the last channel (shadows/landmarks),
        # so instant-reward detection ignores passive visual elements.
        self._sender_color_filter: np.ndarray = self._make_filter(self.board_size, env.n_sender_channels)
        self._receiver_color_filter: np.ndarray = self._make_filter(self.board_size, env.n_receiver_channels)

        self.instant_multiplier: float = instant_multiplier
        self.end_multiplier: float = end_multiplier
        self.shaping_multiplier: float = shaping_multiplier
        self.sender_shaping_multiplier: float = sender_shaping_multiplier
        self.gamma: float = gamma
        self.shaping_gamma: float = gamma if shaping_gamma is None else shaping_gamma
        self.device: str = device

        self.num_moves: int = 0
        self.done: bool = False
        self.perf_epsilon: float = perf_epsilon
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
        cur = cur / 255.0

        # previous_boards: list of H,W,3 -> [1, 3*len, H, W]
        prev_list = [
            torch.as_tensor(b, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
            for b in previous_boards
        ]
        prev = torch.cat(prev_list, dim=1)
        prev = prev / 255.0

        prog = torch.tensor([[progress]], dtype=torch.float32, device=self.device)

        return Observation(current_board=cur, previous_boards=prev, progress=prog)
    
    @staticmethod
    def _make_filter(board_size: int, n_channels: int) -> np.ndarray:
        f = np.ones((board_size, board_size, n_channels), dtype=np.float32)
        f[:, :, -1] = 0.0  # ignore shadow / landmark channel
        return f

    def _end_episode(self, final_reward=None, final_perf=None) -> None:
        self.done = True
        if final_reward is None or final_perf is None:
            final_reward, final_perf = self.env.reward_function()
        self.final_reward = float(final_reward) * self.end_multiplier
        self.final_performance = float(final_perf)

    def _maybe_early_exit_on_perfect_guess(self) -> None:
        r, perf = self.env.reward_function()
        if float(perf) >= 1.0 - self.perf_epsilon:
            self._end_episode(r, perf)

    def _instant_reward(self, action_history: deque[int], board_history: deque[np.ndarray], color_filter: np.ndarray) -> float:
        if False not in (board_history[-1] * color_filter == board_history[-3] * color_filter):
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

    def sender_observe_raw(self) -> tuple[np.ndarray, list[np.ndarray], float]:
        """Returns raw numpy arrays; use batch_observe_to_tensor for vectorized training."""
        return (
            self.env.sender_agent_view(),
            list(self.sender_board_history)[-self.history_len :],
            self.num_moves / self.max_moves,
        )

    def receiver_observe_raw(self) -> tuple[np.ndarray, list[np.ndarray], float]:
        return (
            self.env.receiver_agent_view(),
            list(self.receiver_board_history)[-self.history_len :],
            self.num_moves / self.max_moves,
        )

    @staticmethod
    def batch_observe_to_tensor(
        raw_list: list[tuple[np.ndarray, list[np.ndarray], float]],
        device: str,
    ) -> Observation:
        """Batch-convert N raw observations to a single Observation tensor in one pass."""
        N = len(raw_list)
        cur0, prev0, _ = raw_list[0]
        H, W, C = cur0.shape
        h_len = len(prev0)

        cur_np = np.empty((N, C, H, W), dtype=np.float32)
        prev_np = np.empty((N, C * h_len, H, W), dtype=np.float32)
        prog_np = np.empty((N, 1), dtype=np.float32)

        for i, (cur, prevs, prog) in enumerate(raw_list):
            cur_np[i] = cur.transpose(2, 0, 1)
            for j, p in enumerate(prevs):
                prev_np[i, j * C : (j + 1) * C] = p.transpose(2, 0, 1)
            prog_np[i, 0] = prog

        cur_t = torch.from_numpy(cur_np).to(device) / 255.0
        prev_t = torch.from_numpy(prev_np).to(device) / 255.0
        prog_t = torch.from_numpy(prog_np).to(device)
        return Observation(current_board=cur_t, previous_boards=prev_t, progress=prog_t)
    
    def sender_act(self, action: int) -> tuple[float, bool]:
        if self.done:
            raise RuntimeError("The action limit was exhausted. Reset the environment.")
        self.num_moves += 1

        if self.sender_shaping_multiplier != 0.0:
            pre_clue_dist = self.env.distance_func(self.env.board1_clues, self.env.board1_landmarks)

        self.env.sender_agent_action(action)

        self.sender_board_history.append(self.env.sender_agent_view())
        self.sender_action_history.append(action)

        instant_reward = self._instant_reward(self.sender_action_history, self.sender_board_history, self._sender_color_filter)

        if self.sender_shaping_multiplier != 0.0:
            post_clue_dist = self.env.distance_func(self.env.board1_clues, self.env.board1_landmarks)
            instant_reward += (pre_clue_dist - self.shaping_gamma * post_clue_dist) * self.sender_shaping_multiplier

        if not self.done:
            self._maybe_early_exit_on_perfect_guess()

        if not self.done and self.num_moves >= self.max_moves:
            self._end_episode()

        self.animation_frames.append(self.env.draw_boards())
        return instant_reward, self.done

    def receiver_act(self, action: int) -> tuple[float, bool]:
        if self.done:
            raise RuntimeError("The action limit was exhausted. Reset the environment.")
        self.num_moves += 1

        if self.shaping_multiplier != 0.0:
            pre_dist = self.env.distance_func(self.env.board1_landmarks, self.env.board2_guesses)

        self.env.receiver_agent_action(action)

        self.receiver_board_history.append(self.env.receiver_agent_view())
        self.receiver_action_history.append(action)

        instant_reward = self._instant_reward(self.receiver_action_history, self.receiver_board_history, self._receiver_color_filter)

        if self.shaping_multiplier != 0.0:
            post_dist = self.env.distance_func(self.env.board1_landmarks, self.env.board2_guesses)
            # Proper PBRS: F(s,s') = γΦ(s') - Φ(s) with Φ(s) = -distance
            instant_reward += (pre_dist - self.shaping_gamma * post_dist) * self.shaping_multiplier

        if not self.done:
            self._maybe_early_exit_on_perfect_guess()

        if not self.done and self.num_moves >= self.max_moves:
            self._end_episode()

        self.animation_frames.append(self.env.draw_boards())
        return instant_reward, self.done
    
    def get_useless_action_val(self) -> int:
        return int(self.env.useless_action_flag)

    def get_clue_landmark_distance(self) -> float:
        """Distance from clues to their nearest landmarks at the current moment.
        Converging toward 0 during training means the sender is learning to communicate."""
        return float(self.env.distance_func(self.env.board1_clues, self.env.board1_landmarks))

    def get_final_reward(self) -> float:
        if not self.done or self.final_reward is None:
            raise RuntimeError("The episode is not over yet.")
        return self.final_reward

    def get_final_performance(self) -> float:
        if not self.done or self.final_performance is None:
            raise RuntimeError("The episode is not over yet.")
        return self.final_performance
