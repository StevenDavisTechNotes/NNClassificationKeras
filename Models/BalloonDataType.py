from dataclasses import dataclass

@dataclass(frozen=True)
class Balloon():
    diameter: float
    background_color: int
    foreground_color: int
    shapes: int


@dataclass(frozen=True)
class ClassifiedBalloon():
    balloon: Balloon
    classification: int
