"""
@File    :   ui_main.py
@Time    :   2024/08/16 09:30:00
@Author  :   Zih-Hao (Mike) Huang
@Version :   1.0
@Contact :   main668888@gmail.com
"""

# ! python3
# -*- encoding: utf-8 -*-

import logging
import os
import sys
from types import TracebackType
from typing import Optional

import cv2
import numpy as np
import requests
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from handlers.clear_point import clear_points
from handlers.clickable_label import ClickableLabel, on_image_click
from handlers.load_folder import load_folder
from handlers.mask2img import apply_mask_to_image
from handlers.predict_mask import predict_mask
from handlers.resize_image import resize_image
from handlers.save_mask import save_mask

BASE_PATH = os.getcwd()


class SAMApp(QWidget):
    def __init__(self):
        """
        Initialize the SAMApp.
        """
        super().__init__()
        self.setWindowTitle("SAM2 Model UI")

        self._initialize_members()
        self.setup_logger()
        self.initUI()

        self.check_and_download_model()
        self.initialize_predictor()

    def _initialize_members(self):
        """
        Initialize class members.
        """
        self.image = None
        self.image_list = []
        self.predictor = None
        self.input_points = []
        self.input_labels = []
        self.label_name = None
        self.labels_dict = {}
        self.label_counter = 0

        self.labels_path = "labels.txt"
        self.device = "cuda"
        self.model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        self.sam2_checkpoint = os.path.join(
            BASE_PATH, "checkpoints/sam2.1_hiera_large.pt"
        )
        self.model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

    def setup_logger(self):
        """
        Setup logger configuration.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        file_handler = logging.FileHandler("app.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        sys.excepthook = self.handle_exception

    def handle_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Optional[TracebackType],
    ):
        """
        Handle uncaught exceptions by logging them.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    def initUI(self):
        """
        Initialize the user interface.
        """

        self.setup_image_labels()
        self.setup_buttons()
        self.setup_layouts()

        self.logger.info("UI Initialized.")

    def setup_image_labels(self):
        """
        Setup image labels in the UI.
        """
        self.image_label = ClickableLabel(self)
        self.image_label.parent().on_image_click = self.on_image_click
        self.image_label.setFixedSize(600, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.predicted_label = QLabel()
        self.predicted_label.setFixedSize(600, 600)
        self.predicted_label.setAlignment(Qt.AlignCenter)
        self.predicted_label_mask = QLabel()
        self.predicted_label_mask.setFixedSize(600, 600)
        self.predicted_label_mask.setAlignment(Qt.AlignCenter)

    def setup_buttons(self):
        """
        Setup buttons in the UI.
        """
        self.load_button = QPushButton("Load Folder", self)
        self.load_button.clicked.connect(lambda: load_folder(self))

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Enter Label Name")
        self.label_input.textChanged.connect(self.update_label_name)

        self.predict_button = QPushButton("Predict Mask", self)
        self.predict_button.clicked.connect(lambda: predict_mask(self))

        self.save_mask_button = QPushButton("Save Mask", self)
        self.save_mask_button.clicked.connect(lambda: save_mask(self))

        self.clear_points_button = QPushButton("Clear Points", self)
        self.clear_points_button.clicked.connect(lambda: clear_points(self))

    def setup_layouts(self):
        """
        Setup layouts for main UI components.
        """
        main_layout = QGridLayout()
        main_layout.addWidget(self.image_label, 0, 0)
        main_layout.addWidget(self.predicted_label, 0, 1)
        main_layout.addWidget(self.predicted_label_mask, 0, 2)

        side_layout = QVBoxLayout()
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentItemChanged.connect(self.display_image)
        side_layout.addWidget(self.image_list_widget)

        for button in [
            self.load_button,
            self.label_input,
            self.predict_button,
            self.save_mask_button,
            self.clear_points_button,
        ]:
            side_layout.addWidget(button)

        main_layout.addLayout(side_layout, 1, 0, 1, 2)
        self.setLayout(main_layout)

    def check_and_download_model(self):
        """
        Check if the model file exists and download if not.
        """
        checkpoints_dir = os.path.dirname(self.sam2_checkpoint)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        if not os.path.exists(self.sam2_checkpoint):
            self.download_model()

    def download_model(self):
        """
        Download the model if it does not exist.
        """
        reply = QMessageBox.question(
            self,
            "Model Download",
            "Model file not found. Do you want to download it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.logger.info("Downloading model...")
            try:
                response = requests.get(self.model_url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                progress_dialog = QProgressDialog(
                    "Downloading model...", "Cancel", 0, total_size // 1024, self
                )
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setMinimumDuration(0)
                progress_dialog.show()

                downloaded_size = 0
                with open(self.sam2_checkpoint, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if progress_dialog.wasCanceled():
                            self.logger.info("Model download canceled by user.")
                            os.remove(self.sam2_checkpoint)
                            return
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress_dialog.setValue(downloaded_size // 1024)

                self.logger.info("Model downloaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to download the model: {e}")
                QMessageBox.critical(
                    self, "Download Failed", f"Failed to download the model: {e}"
                )
                raise
        else:
            self.logger.info("Model download canceled by user.")
            sys.exit()

    def initialize_predictor(self):
        """
        Initialize the SAM predictor model.
        """
        try:
            self.sam2 = build_sam2(
                self.model_cfg, self.sam2_checkpoint, device=self.device
            )
            self.sam2.to(device=self.device)
            self.logger.info("Model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")

    def closeEvent(self, event):
        """
        Handle the close event by logging a message.
        """
        self.logger.info("Close app " + "=" * 40)
        event.accept()

    def update_label_name(self, text: str):
        """
        Update label.
        """
        self.label_name: str = text.strip()

    def display_image(self):
        """
        Display the selected image.
        """
        item: Optional[QListWidgetItem] = self.image_list_widget.currentItem()
        if item:
            self.image_path: str = item.data(Qt.UserRole)
            if self.image_path:
                self.image: np.ndarray = cv2.imread(self.image_path)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

                resized_image: np.ndarray = resize_image(
                    self.image,
                    label_width=self.image_label.width(),
                    label_height=self.image_label.height(),
                )

                height, width, _ = resized_image.shape
                bytes_per_line: int = 3 * width
                q_image: QImage = QImage(
                    resized_image.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )
                self.image_label.setPixmap(QPixmap.fromImage(q_image))

                self.predictor: SAM2ImagePredictor = SAM2ImagePredictor(self.sam2)
                self.predictor.set_image(self.image)
                self.input_points = []
                self.input_labels = []
                self.image_label.clicked_position = []
                self.image_label.update()

    def on_image_click(self, x: int, y: int, label: int):
        """
        Handle image click events.

        :param x: The x-coordinate of the click.
        :param y: The y-coordinate of the click.
        :param label: The label associated with the click (1 for left click, 0 for right click).
        """
        on_image_click(self, x, y, label)

    def show_selected_mask(self, mask: np.ndarray, label_key: int):
        """
        Show the selected mask.

        :param mask: The selected mask.
        :param label_key: The label key to determine the mask color.
        """
        self.mask_image, self.label_image = apply_mask_to_image(
            self.image, mask, label_key
        )

        mask_resize_image: np.ndarray = resize_image(
            self.mask_image,
            label_width=self.image_label.width(),
            label_height=self.image_label.height(),
        )
        label_resize_image: np.ndarray = resize_image(
            self.label_image,
            label_width=self.image_label.width(),
            label_height=self.image_label.height(),
        )

        height, width, _ = mask_resize_image.shape
        bytes_per_line: int = 3 * width

        q_image: QImage = QImage(
            mask_resize_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.predicted_label.setPixmap(QPixmap.fromImage(q_image))
        q_image: QImage = QImage(
            label_resize_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.predicted_label_mask.setPixmap(QPixmap.fromImage(q_image))

    def save_mask(self):
        """
        Save the predicted mask to a file.
        """
        mask_folder_path: str = os.path.join(os.path.dirname(self.folder_path), "masks")
        if not os.path.exists(mask_folder_path):
            os.makedirs(mask_folder_path)

        image_path: str = self.image_path.split("\\")[-1]
        image_path = image_path.split(".")[0]
        image_name_without_extension: str = os.path.splitext(image_path)[0]
        image_name_without_extension = image_name_without_extension.rsplit(".", 1)[0]
        status: bool = cv2.imwrite(
            f"{mask_folder_path}/{image_name_without_extension}_mask.png",
            cv2.cvtColor(self.label_image, cv2.COLOR_RGB2BGR),
        )
        if status:
            self.logger.info(
                f"Mask saved: {mask_folder_path}/{image_name_without_extension}_mask.png"
            )
        else:
            self.logger.error(
                f"Failed to save mask: {mask_folder_path}/{image_name_without_extension}_mask.png"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = SAMApp()
    ex.show()
    sys.exit(app.exec_())
