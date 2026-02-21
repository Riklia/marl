from dataclasses import dataclass
import torch

@dataclass(slots=True)
class Observation:
    current_board: torch.Tensor
    previous_boards: torch.Tensor
    progress: torch.Tensor
