import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from handlers.mask2img import apply_mask_to_image
from handlers.resize_image import resize_image


class MaskInfo:
    def __init__(self, masks: np.ndarray, scores: list[float]):
        self.masks = masks
        self.scores = scores


class MaskSelectionDialog(QDialog):
    """Sub window for mask."""

    def __init__(self, image: np.ndarray, mask_info: MaskInfo, label_key: int):
        """
        Initialize the MaskSelectionDialog.

        :param image: The original image.
        :param masks: List of mask arrays.
        :param scores: List of scores corresponding to each mask.
        :param label_key: The label key to determine the mask color.
        """
        super().__init__()
        self.setWindowTitle("Select Mask")

        self.image: np.ndarray = image
        self.mask_info: MaskInfo = mask_info
        self.label_key: int = label_key
        self.current_index: int = 0

        self.initUI()
        self.show_mask(self.current_index)

    def initUI(self):
        """Initialize the user interface components."""
        self.layout = QVBoxLayout()

        self.image_label: QLabel = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.score_label: QLabel = QLabel()
        self.score_label.setAlignment(Qt.AlignCenter)
        font: QFont = QFont()
        font.setPointSize(16)
        self.score_label.setFont(font)
        self.layout.addWidget(self.score_label)

        self.num_masks_label: QLabel = QLabel()
        self.num_masks_label.setAlignment(Qt.AlignCenter)
        font: QFont = QFont()
        font.setPointSize(16)
        self.num_masks_label.setFont(font)
        self.layout.addWidget(self.num_masks_label)

        self.update_info()

        self.button_layout: QHBoxLayout = QHBoxLayout()

        self.prev_button: QPushButton = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_mask)
        self.button_layout.addWidget(self.prev_button)

        self.next_button: QPushButton = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_mask)
        self.button_layout.addWidget(self.next_button)

        self.select_button: QPushButton = QPushButton("Select")
        self.select_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.select_button)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def update_info(self):
        """Update the score and mask count information labels."""
        if not hasattr(self, "sorted_ind"):
            self.sorted_ind = np.argsort(self.mask_info.scores)[::-1]

        if 0 <= self.current_index < len(self.mask_info.masks):
            score = self.mask_info.scores[self.sorted_ind][self.current_index]
            self.score_label.setText(f"Score: {score:.3f}")
            self.num_masks_label.setText(
                f"Masks: {self.current_index + 1} / {len(self.mask_info.masks)}"
            )
        else:
            self.score_label.setText("Score: N/A")
            self.num_masks_label.setText(f"Masks: 0 / {len(self.mask_info.masks)}")

    def show_mask(self, index: int):
        """
        Display the mask at the specified index.

        :param index: Index of the mask to be displayed.
        """
        mask_image, _ = apply_mask_to_image(
            self.image, self.mask_info.masks[index], self.label_key
        )
        mask_image = resize_image(
            mask_image,
            label_width=self.image_label.width(),
            label_height=self.image_label.height(),
        )
        height, width, _ = mask_image.shape
        bytes_per_line: int = 3 * width
        q_image: QImage = QImage(
            mask_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        self.image_label.adjustSize()

        self.update_info()

    def show_previous_mask(self):
        """Show the previous mask in the list."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_mask(self.current_index)

    def show_next_mask(self):
        """Show the next mask in the list."""
        if self.current_index < len(self.mask_info.masks) - 1:
            self.current_index += 1
            self.show_mask(self.current_index)

    def get_selected_mask_index(self) -> int:
        """
        Get the index of the currently selected mask.

        :return: The index of the selected mask.
        """
        return self.current_index
