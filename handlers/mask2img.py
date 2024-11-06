import hashlib
from typing import Tuple

import numpy as np


def apply_mask_to_image(
    image: np.ndarray, mask: np.ndarray, label_key: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the mask to the image with the specified color.

    :param image: The original image.
    :param mask: The mask to be applied.
    :param label_key: The label key to determine the mask color.
    :return: The image with the mask applied.
    """
    color = get_color_from_label_key(label_key)
    masked_image: np.ndarray = image.copy()
    label_image: np.ndarray = np.zeros_like(image)
    for channel in range(3):
        masked_image[:, :, channel] = np.where(
            mask,
            masked_image[:, :, channel] * 0.3 + color[channel] * 0.7,
            masked_image[:, :, channel],
        )
        label_image[:, :, channel] = np.where(
            mask, color[channel], label_image[:, :, channel]
        )
    return masked_image, label_image


def get_color_from_label_key(label_key: int) -> list[int]:
    """
    Generate a color based on the label key.

    :param label_key: The label key.
    :return: A list of RGB values.
    """
    hash_object = hashlib.md5(str(label_key).encode())
    hex_dig = hash_object.hexdigest()
    color_r = int(hex_dig[0:2], 16)
    color_g = int(hex_dig[2:4], 16)
    color_b = int(hex_dig[4:6], 16)
    return [color_r, color_g, color_b]
