import cv2
import numpy as np


def resize_image(image: np.ndarray, label_width: int, label_height: int) -> np.ndarray:
    image_height, image_width, _ = image.shape
    image_aspect_ratio: float = image_width / image_height
    label_aspect_ratio: float = label_width / label_height

    if image_aspect_ratio > label_aspect_ratio:
        new_width: int = label_width
        new_height: int = int(new_width / image_aspect_ratio)
    else:
        new_height: int = label_height
        new_width: int = int(new_height * image_aspect_ratio)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
