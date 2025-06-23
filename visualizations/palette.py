from typing import List
import numpy as np


def palette_val(palette: List[tuple]) -> List[tuple]:
    """Convert palette to matplotlib palette.

    Args:
        palette (List[tuple]): A list of color tuples.

    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color = [c / 255 for c in color]
        new_palette.append(tuple(color))
    return new_palette


def jitter_color(color: tuple) -> tuple:
    """
    Randomly jitter the given color in the range [0, 1] to better distinguish instances
    with the same class.

    Args:
        color (tuple): The RGB color tuple. Each value is between [0, 1].

    Returns:
        tuple: The jittered color tuple.
    """
    jitter = (np.random.rand(3) - 0.5) * 0.6
    color = np.clip(np.array(color) + jitter, 0, 1)
    return tuple(color)


def random_color():
    color = np.random.random((1, 3)).tolist()[0]
    return color
