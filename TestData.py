import random
from typing import List
from Models.ArgumentsDataType import Arguments
from Models.BalloonDataType import Balloon, ClassifiedBalloon
from Utils import always_true


def generate_data(args: Arguments,
                  num_samples: int) -> List[ClassifiedBalloon]:
    def perturbed(i_sample: int, range: int) -> int:
        val = i_sample % range
        if random.random() < args.noise_level:
            return max(
                0, min(range - 1, (val + (-1 + 2 * random.randint(0, 1)))))
        else:
            return val
    characteristic_counts = [args.num_backgrounds, args.num_foregrounds, args.num_shapes]
    num_categories = max(characteristic_counts)
    for ccount in characteristic_counts:
        assert ccount <= num_categories and num_categories % ccount == 0

    results = [
        ClassifiedBalloon(
            balloon=Balloon(
                diameter=random.random(),
                background_color=perturbed(i_sample, args.num_backgrounds),
                foreground_color=perturbed(i_sample, args.num_foregrounds),
                shapes=perturbed(i_sample, args.num_shapes),
            ),
            classification= i_sample % num_categories,
        )
        for i_sample in range(num_samples)
    ]
    if args.shuffle is True:
        random.shuffle(results)
    return results
