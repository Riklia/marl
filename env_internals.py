import numpy as np
import matplotlib.pyplot as plt
from misc_utils import greedy_distance

from custom_types import Action, Move, ActionType


class BoardsImplementation:
    def __init__(self, size: int, n_landmarks: int, n_clues: int, n_questions: int, linked_shadows: bool = True, seed: int | None = None):
        if size < 1:
            raise ValueError("The board size should be positive.")
        if n_landmarks < 1:
            raise ValueError("The number of landmarks should be positive.")
        if n_clues < 1:
            raise ValueError("The number of clues should be positive.")
        if n_questions < 0:
            raise ValueError("The number of questions should not be negative.")
        if n_landmarks + n_clues + n_questions > size ** 2 / 2:
            raise ValueError("The boards should have more empty space on them.")
        if seed is None:
            seed = np.random.randint(42, 45954)
        elif seed < 0:
            raise ValueError("The seed should be non negative.")
        
        self.rng = np.random.default_rng(seed) # Creating an rng engine for consistency in board generation. 
        # Not guaranteed to be consistent between different versions of numpy.
        self.size = size # Size of the boards. They are the same shape and are both square, so only a single number is required.
        self.n_landmarks = n_landmarks # Number of landmark objects on board 1 whose position is to be guessed by the agent looking at board 2.
        # Board 2 has guess objects used for that. Their number is equal to the number of landmarks on board 1.
        self.n_clues = n_clues # Number of clue objects on board 1. Each of them casts a shadow onto board 2.
        self.n_questions = n_questions # Number of question objects on board 2. Each of them casts a shadow onto board 1.
        self.linked_shadows = linked_shadows # Should shadows being cast by clue and question objects move with them? 
        # Passing in False results in shadows being frozen in their initial position.
        self.distance_func = greedy_distance

        # None is "do nothing" action
        self.sender_agent_actions = [None]
        for i in range(n_clues):
            self.sender_agent_actions.extend([
                Action(ActionType.CLUE, i, Move(0, -1)),
                Action(ActionType.CLUE, i, Move(0, 1)),
                Action(ActionType.CLUE, i, Move(-1, 0)),
                Action(ActionType.CLUE, i, Move(1, 0)),
            ])

        self.receiver_agent_actions = [None]
        for i in range(n_landmarks):
            self.receiver_agent_actions.extend([
                Action(ActionType.GUESS, i, Move(0, -1)),
                Action(ActionType.GUESS, i, Move(0, 1)),
                Action(ActionType.GUESS, i, Move(-1, 0)),
                Action(ActionType.GUESS, i, Move(1, 0)),
            ])

        for i in range(n_questions):
            self.receiver_agent_actions.extend([
                Action(ActionType.QUESTION, i, Move(0, -1)),
                Action(ActionType.QUESTION, i, Move(0, 1)),
                Action(ActionType.QUESTION, i, Move(-1, 0)),
                Action(ActionType.QUESTION, i, Move(1, 0)),
            ])

        self._calculate_neutral_distance()
        self.populate_boards() # Assign positions to objects on the boards.
    
    def _calculate_neutral_distance(self): # Not meant to be used outside of the class.
        private_rng = np.random.default_rng(self.size) # We want the same number for all boards of a given size and number of landmarks.
        
        distances = list()
        n = 1000
        
        for _ in range(n): # Averaging distances from n board samples and using that as a neutral distance.
            all_coordinates = [(x, y) for x in range(self.size) for y in range(self.size)]
            board1_space_picks = [(int(i[0]), int(i[1])) for i in private_rng.permutation(all_coordinates)]
            board2_space_picks = [(int(i[0]), int(i[1])) for i in private_rng.permutation(all_coordinates)]
            board1_landmarks = board1_space_picks[:self.n_landmarks]
            board2_guesses = board2_space_picks[:self.n_landmarks]
            distance = self.distance_func(board1_landmarks, board2_guesses)
            distances.append(distance)
            
        self.neutral_distance = max(sum(distances) / n, 0.5)

    def _try_move_in_list(
            self,
            moving: list[tuple[int, int]],
            object_number: int,
            move: Move,
            *,
            blocked_by: list[tuple[int, int]] = (),
    ) -> None:
        old_position = moving[object_number]
        new_position = (old_position[0] + move.dx, old_position[1] + move.dy)

        if not (0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size):
            self.useless_action_flag = True
            return

        if new_position in blocked_by:
            self.useless_action_flag = True
            return

        if new_position in moving and new_position != old_position:
            self.useless_action_flag = True
            return

        moving[object_number] = new_position
    
    def populate_boards(self): # Could also be used to reset the environment to a random state.
        # Create mixed up lists of coordinates for both boards to get non repeating positions for random placements of the objects.
        all_coordinates = [(x, y) for x in range(self.size) for y in range(self.size)]
        board1_space_picks = [(int(i[0]), int(i[1])) for i in self.rng.permutation(all_coordinates)]
        board2_space_picks = [(int(i[0]), int(i[1])) for i in self.rng.permutation(all_coordinates)]
        
        # Establish positions of landmarks on board 1.
        self.board1_landmarks = board1_space_picks[:self.n_landmarks]
        board1_space_picks = board1_space_picks[self.n_landmarks:] # Take out the used positions from the pool.
        self.board1_landmarks.sort()
        
        # Establish positions of guesses on board 2
        self.board2_guesses = board2_space_picks[:self.n_landmarks]
        board2_space_picks = board2_space_picks[self.n_landmarks:] # Take out the used positions from the pool.
        self.board2_guesses.sort()
        
        # Establish positions of clues on board 1.
        self.board1_clues = board1_space_picks[:self.n_clues]
        # board1_space_picks = board1_space_picks[self.n_clues:] # Not needed.
        self.board1_clues.sort()
        self.board2_c_shadows = self.board1_clues.copy() # Used only if shadows are frozen.
        
        # Establish positions of questions on board 2.
        self.board2_questions = board2_space_picks[:self.n_questions]
        # board2_space_picks = board2_space_picks[self.n_questions:] # Not needed.
        self.board2_questions.sort()
        self.board1_q_shadows = self.board2_questions.copy() # Used only if shadows are frozen.
        
        self.start_distance = max(self.distance_func(self.board1_landmarks, self.board2_guesses), 0.5)
        self.useless_action_flag = False
        
    def sender_agent_view(self): # Sender agent can only see board 1.
        board1_img = np.zeros((self.size, self.size, 3), dtype = np.uint8)
        for x, y in self.board1_landmarks:
            board1_img[y, x, 0] = 255 # Landmarks on channel 0. (Red)
        for x, y in self.board1_clues:
            board1_img[y, x, 1] = 255 # Clues on channel 1. (Green)
        for x, y in (self.board2_questions if self.linked_shadows else self.board1_q_shadows):
            # If the shadows are linked we can just use the positions of objects generating them on the other board.
            board1_img[y, x, 2] = 255 # Shadows on channel 2. (Blue)
            # Shadows can overlap with other objects.
        return board1_img
    
    def receiver_agent_view(self): # Receiver agent can only see board 2.
        board2_img = np.zeros((self.size, self.size, 3), dtype = np.uint8)
        for x, y in self.board2_guesses:
            board2_img[y, x, 0] = 255 # Guesses on channel 0. (Red)
        for x, y in self.board2_questions:
            board2_img[y, x, 1] = 255 # Questions on channel 1. (Green)
        for x, y in (self.board1_clues if self.linked_shadows else self.board2_c_shadows):
            # If the shadows are linked we can just use the positions of objects generating them on the other board.
            board2_img[y, x, 2] = 255 # Shadows on channel 2. (Blue)
            # Shadows can overlap with other objects.
        return board2_img

    def sender_agent_action(self, action_index: int) -> None:
        if action_index < 0 or action_index >= len(self.sender_agent_actions):
            raise ValueError(f"Sender agent does not have action index {action_index}.")

        action = self.sender_agent_actions[action_index]
        self.useless_action_flag = False
        if action is None:
            self.useless_action_flag = True
            return

        if action.object_type != ActionType.CLUE:
            raise RuntimeError(f"Unexpected object_type for sender: {action.object_type}")

        self._try_move_in_list(
            self.board1_clues,
            action.object_number,
            action.move,
            blocked_by=self.board1_landmarks,
        )

    def receiver_agent_action(self, action_index: int) -> None:
        if action_index < 0 or action_index >= len(self.receiver_agent_actions):
            raise ValueError(f"Receiver agent does not have an action with index {action_index}.")
        action = self.receiver_agent_actions[action_index]
        self.useless_action_flag = False
        if action is None:
            self.useless_action_flag = True
            return

        if action.object_type == ActionType.QUESTION:
            self._try_move_in_list(
                self.board2_questions,
                action.object_number,
                action.move,
                blocked_by=self.board2_guesses,
            )
        elif action.object_type == ActionType.GUESS:
            self._try_move_in_list(
                self.board2_guesses,
                action.object_number,
                action.move,
                blocked_by=self.board2_questions,
            )
        else:
            raise RuntimeError(f"Unexpected object_type: {action.object_type}")

    def draw_boards(self):
        """Returns the current state of the boards as a nice, low-quality image with a boring, gray frame."""
        image = np.concatenate((
            np.zeros((1, self.size * 2 + 3, 3), dtype = np.uint8) + 100,
            np.concatenate((
                np.zeros((self.size, 1, 3), dtype = np.uint8) + 100,
                self.sender_agent_view(),
                np.zeros((self.size, 1, 3), dtype = np.uint8) + 100,
                self.receiver_agent_view(),
                np.zeros((self.size, 1, 3), dtype = np.uint8) + 100
            ), axis = 1),
            np.zeros((1, self.size * 2 + 3, 3), dtype = np.uint8) + 100
        ), axis = 0)
        return image
    
    def show_boards(self):
        image = self.draw_boards()
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def reward_function(self):
        """
        Computes the final episode reward and diagnostic performance metric.

        Reward is normalized using the precomputed neutral distance to ensure stability across episodes, which
        is preferred for agents. Performance is normalized by the initial distance and is intended for
        evaluation/analysis only.
        """
        current_distance = self.distance_func(self.board1_landmarks, self.board2_guesses)
        reward = 1.0 - current_distance / self.neutral_distance
        performance = 1.0 - current_distance / self.start_distance
        return reward, performance