#!python3
# Usage: python Runner.py --learning-size 100 --network-depth 3 --network-width 32 --num-backgrounds 10 --num-shapes 10 --num-contrasts 10 --random-seed 1234 --verification-size 20 --no-shuffle
import argparse
from typing import List, Optional

from Models.ArgumentsDataType import Arguments
from TestData import generate_data
import TrainModel

from Utils import set_random_seed

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEBUG_ARGS: Optional[List[str]] = None if False else (
    []
    + '--learning-size 100'.split()
    + '--num-backgrounds 10'.split()
    + '--num-shapes 5'.split()
    + '--num-contrasts 5'.split()
    + '--random-seed 1234'.split()
    + '--verification-size 20'.split()
    + '--no-shuffle'.split()
)


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-size', type=int, default=20)
    parser.add_argument('--num-backgrounds', type=int, default=1)
    parser.add_argument('--num-shapes', type=int, default=1)
    parser.add_argument('--num-contrasts', type=int, default=10)
    parser.add_argument('--random-seed', type=int)
    parser.add_argument('--verification-size', type=int, default=20)
    parser.add_argument('--background-weight', type=float, default=3.0)
    parser.add_argument('--foreground-weight', type=float, default=2.0)
    parser.add_argument('--shape-weight', type=float, default=1.0)
    parser.add_argument('--noise-level', type=float, default=0.2)
    parser.add_argument(
        '--shuffle',
        default=True,
        action=argparse.BooleanOptionalAction)
    if DEBUG_ARGS is None:  # pyright: ignore[reportUnnecessaryComparison]
        args = parser.parse_args()
    else:
        args = parser.parse_args(DEBUG_ARGS)
    return Arguments(
        learning_size=args.learning_size,
        num_backgrounds=args.num_backgrounds,
        num_shapes=args.num_shapes,
        num_foregrounds=args.num_contrasts,
        random_seed=args.random_seed,
        verification_size=args.verification_size,
        background_weight=args.background_weight,
        foreground_weight=args.foreground_weight,
        shape_weight=args.shape_weight,
        noise_level=args.noise_level,
        shuffle=args.shuffle,
    )


def main():
    args = parse_args()
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    learning_data = generate_data(args, args.learning_size)
    verification_data = generate_data(args, args.verification_size)
    TrainModel.train_and_validate6(args, learning_data, verification_data)


if __name__ == "__main__":
    main()
