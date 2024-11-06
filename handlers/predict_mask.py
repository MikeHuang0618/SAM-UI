import os

import numpy as np
from PyQt5.QtWidgets import QDialog, QMessageBox

from select_page import MaskInfo, MaskSelectionDialog


def predict_mask(app):
    """
    Predict the mask for the selected points.
    """
    if app.image is not None and app.input_points:
        input_points_np = np.array(app.input_points)
        input_labels_np = np.array(app.input_labels)
        app.logger.info(f"Points: {app.input_points}\nLabels: {app.input_labels}")
        masks, scores, _ = app.predictor.predict(
            point_coords=input_points_np,
            point_labels=input_labels_np,
            multimask_output=True,
        )

        if app.label_name:
            label_key = add_label(app, app.label_name)
        else:
            reply = QMessageBox.question(
                app,
                "Enter label name",
                "You did not enter a label name",
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                return

        show_mask_selection_dialog(app, masks, scores, label_key)

        app.logger.info(f"Predicted mask for image: {app.image_path}")


def show_mask_selection_dialog(app, masks, scores, label_key):
    """
    Show the dialog to select a mask.

    :param masks: The predicted masks.
    :param scores: The scores of the masks.
    :param label_key: The label key to determine the mask color.
    """
    mask_info = MaskInfo(masks, scores)
    dialog = MaskSelectionDialog(app.image, mask_info, label_key)

    if dialog.exec_() == QDialog.Accepted:
        selected_mask_index = dialog.get_selected_mask_index()
        app.show_selected_mask(masks[selected_mask_index], label_key)


def add_label(app, label_name: str) -> int:
    """
    Added label and return key.
    """
    for key, value in app.labels_dict.items():
        if value == label_name:
            return key

    new_key = app.label_counter
    app.labels_dict[new_key] = label_name
    app.label_counter += 1

    with open(os.path.join(app.folder_path, app.labels_path), "w") as file:
        for key, value in app.labels_dict.items():
            file.write(f"{key}: {value}\n")

    return new_key
