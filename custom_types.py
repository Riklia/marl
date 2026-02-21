import torch
from dataclasses import dataclass
from enum import Enum


@dataclass(slots=True)
class Observation:
    current_board: torch.Tensor
    previous_boards: torch.Tensor
    progress: torch.Tensor

@dataclass(frozen=True)
class Move:
    dx: int
    dy: int

class ActionType(Enum):
    CLUE = "clue"
    GUESS = "guess"
    QUESTION = "question"

@dataclass(frozen=True)
class Action:
    object_type: ActionType
    object_number: int
    move: Move