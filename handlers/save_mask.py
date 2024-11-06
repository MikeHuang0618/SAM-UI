import os

import cv2


def save_mask(sam_ui):
    """
    Save the predicted mask to a file.
    """
    mask_folder_path: str = os.path.join(os.path.dirname(sam_ui.folder_path), "masks")
    if not os.path.exists(mask_folder_path):
        os.makedirs(mask_folder_path)

    image_path: str = sam_ui.image_path.split("\\")[-1]
    image_path = image_path.split(".")[0]
    image_name_without_extension: str = os.path.splitext(image_path)[0]
    image_name_without_extension = image_name_without_extension.rsplit(".", 1)[0]
    status: bool = cv2.imwrite(
        f"{mask_folder_path}/{image_name_without_extension}_mask.png",
        cv2.cvtColor(sam_ui.label_image, cv2.COLOR_RGB2BGR),
    )
    if status:
        sam_ui.logger.info(
            f"Mask saved: {mask_folder_path}/{image_name_without_extension}_mask.png"
        )
    else:
        sam_ui.logger.error(
            f"Failed to save mask: {mask_folder_path}/{image_name_without_extension}_mask.png"
        )
