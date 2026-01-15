import numpy as np
import matplotlib.pyplot as plt
from misc_utils import greedy_distance


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
        
        self.sender_agent_actions = list()
        self.sender_agent_actions.append(None) # Do nothing action.
        for i in range(n_clues): # Sender agent can only move clue objects on board 1.
            self.sender_agent_actions.append(("clue", i, np.array([0, -1]))) # Up
            self.sender_agent_actions.append(("clue", i, np.array([0, 1]))) # Down
            self.sender_agent_actions.append(("clue", i, np.array([-1, 0]))) # Left
            self.sender_agent_actions.append(("clue", i, np.array([1, 0]))) # Right
            
        self.receiver_agent_actions = list()
        self.receiver_agent_actions.append(None) # Do nothing action.
        for i in range(n_landmarks): # Receiver agent can move guess objects on board 2.
            self.receiver_agent_actions.append(("guess", i, np.array([0, -1]))) # Up
            self.receiver_agent_actions.append(("guess", i, np.array([0, 1]))) # Down
            self.receiver_agent_actions.append(("guess", i, np.array([-1, 0]))) # Left
            self.receiver_agent_actions.append(("guess", i, np.array([1, 0]))) # Right
        for i in range(n_questions): # Receiver agent can also move question objects on board 2.
            self.receiver_agent_actions.append(("question", i, np.array([0, -1]))) # Up
            self.receiver_agent_actions.append(("question", i, np.array([0, 1]))) # Down
            self.receiver_agent_actions.append(("question", i, np.array([-1, 0]))) # Left
            self.receiver_agent_actions.append(("question", i, np.array([1, 0]))) # Right
        
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
    
    def sender_agent_action(self, action_index: int):
        if action_index < 0 or action_index >= len(self.sender_agent_actions):
            raise ValueError(f"Sender agent does not have an action with index: {action_index}.")
        action = self.sender_agent_actions[action_index]
        self.useless_action_flag = False
        if action is None:
            self.useless_action_flag = True
            return
        object_type, object_number, move_direction = action
        if object_type == "clue":
            old_position = self.board1_clues[object_number]
            new_position = tuple(old_position + move_direction)
            if new_position[0] < 0 or new_position[0] >= self.size or new_position[1] < 0 or new_position[1] >= self.size:
                self.useless_action_flag = True
                return # Avoiding getting off the board.
            if new_position in self.board1_landmarks or new_position in self.board1_clues:
                self.useless_action_flag = True
                return # Collision avoidance.
            # All returns until now didn't change anything on the boards.
            # That's because objects can't be pushed off the board or into other non-shadow objects.
            self.board1_clues.pop(object_number)
            self.board1_clues.append(new_position)
            self.board1_clues.sort() # Unsure about that.
            # It makes the control scheme not dependant on previous actions, but changes which actions correspond to which objects.
            
    def receiver_agent_action(self, action_index: int):
        if action_index < 0 or action_index >= len(self.receiver_agent_actions):
            raise ValueError(f"Receiver agent does not have an action with index {action_index}.")
        action = self.receiver_agent_actions[action_index]
        self.useless_action_flag = False
        if action is None:
            self.useless_action_flag = True
            return
        object_type, object_number, move_direction = action
        if object_type == "question":
            old_position = self.board2_questions[object_number]
            new_position = tuple(old_position + move_direction)
            if new_position[0] < 0 or new_position[0] >= self.size or new_position[1] < 0 or new_position[1] >= self.size:
                self.useless_action_flag = True
                return # Avoiding getting off the board.
            if new_position in self.board2_guesses or new_position in self.board2_questions:
                self.useless_action_flag = True
                return # Collision avoidance.
            # All returns until now didn't change anything on the boards.
            # That's because objects can't be pushed off the board or into other non-shadow objects.
            self.board2_questions.pop(object_number)
            self.board2_questions.append(new_position)
            self.board2_questions.sort() # Unsure about that.
            # It makes the control scheme not dependant on previous actions, but changes which actions correspond to which objects.
        elif object_type == "guess":
            old_position = self.board2_guesses[object_number]
            new_position = tuple(old_position + move_direction)
            if new_position[0] < 0 or new_position[0] >= self.size or new_position[1] < 0 or new_position[1] >= self.size:
                self.useless_action_flag = True
                return # Avoiding getting off the board.
            if new_position in self.board2_guesses or new_position in self.board2_questions:
                self.useless_action_flag = True
                return # Collision avoidance.
            # All returns until now didn't change anything on the boards.
            # That's because objects can't be pushed off the board or into other non-shadow objects.
            self.board2_guesses.pop(object_number)
            self.board2_guesses.append(new_position)
            self.board2_guesses.sort() # Unsure about that.
            # It makes the control scheme not dependant on previous actions, but changes which actions correspond to which objects.

    def draw_boards(self): # Show the current state of the boards as a nice, low-quality image with a boring, gray frame.
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
    
    def reward_function(self): # The reward function is the same for both agents and should only be calculated and applied at the end of an episode.
        current_distance = self.distance_func(self.board1_landmarks, self.board2_guesses) # The greedy_distance function is expensive to calculate.
        # More stable and suitable as a reward since neutral distance is the same between episodes. Meant for the agents.
        reward = 1.0 - current_distance / self.neutral_distance
        # Allows for easily telling if agents improved or worsened the alignment of landmarks and guesses in current episode. Meant for humans.
        performance = 1.0 - current_distance / self.start_distance
        return reward, performance
        