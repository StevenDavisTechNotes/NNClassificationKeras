

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Arguments:
    background_weight: float
    foreground_weight: float
    shape_weight: float
    noise_level: float
    learning_size: int
    num_backgrounds: int
    num_foregrounds: int
    num_shapes: int
    random_seed: Optional[int]
    verification_size: int
    shuffle: bool
