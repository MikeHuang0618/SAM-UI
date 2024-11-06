# folder_loader.py

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem


def load_folder(sam_ui):
    options = QFileDialog.Options()
    folder_path = QFileDialog.getExistingDirectory(
        sam_ui, "Open Image Folder", options=options
    )
    if folder_path:
        sam_ui.folder_path = folder_path
        sam_ui.image_list = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(("png", "jpg", "jpeg", ".bmp"))
        ]
        sam_ui.image_list_widget.clear()
        for image_path in sam_ui.image_list:
            item = QListWidgetItem(os.path.basename(image_path))
            item.setData(Qt.UserRole, image_path)
            sam_ui.image_list_widget.addItem(item)

        sam_ui.logger.info(
            f"Loaded folder: {folder_path} with {len(sam_ui.image_list)} images."
        )

    if os.path.exists(os.path.join(sam_ui.folder_path, sam_ui.labels_path)):
        with open(os.path.join(sam_ui.folder_path, sam_ui.labels_path), "r") as file:
            for line in file:
                key, value = line.strip().split(": ")
                sam_ui.labels_dict[int(key)] = value
                sam_ui.label_counter = max(sam_ui.label_counter, int(key) + 1)
    else:
        try:
            open(os.path.join(sam_ui.folder_path, sam_ui.labels_path), "a").close()
            print(
                f"File '{os.path.join(sam_ui.folder_path, sam_ui.labels_path)}' created successfully."
            )
        except Exception as e:
            print(
                f"Failed to create file '{os.path.join(sam_ui.folder_path, sam_ui.labels_path)}': {e}"
            )
