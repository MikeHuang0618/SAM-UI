import sys
import requests
import os
import logging
import hashlib
from types import TracebackType
from typing import Optional, Tuple
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QListWidget, QListWidgetItem, QGridLayout, QDialog,
                             QLineEdit, QMessageBox, QProgressDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QFont
from PyQt5.QtCore import Qt
from segment_anything import SamPredictor, sam_model_registry

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, label_key: int) -> Tuple[np.ndarray, np.ndarray]:
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
    for c in range(3):
        masked_image[:, :, c] = np.where(mask, masked_image[:, :, c] * 0.3 + color[c] * 0.7, masked_image[:, :, c])
        label_image[:, :, c] = np.where(mask, color[c], label_image[:, :, c])
    return masked_image, label_image

def get_color_from_label_key(label_key: int) -> list[int]:
    """
    Generate a color based on the label key.

    :param label_key: The label key.
    :return: A list of RGB values.
    """
    hash_object = hashlib.md5(str(label_key).encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    return [r, g, b]

class MaskSelectionDialog(QDialog):
    def __init__(self, image: np.ndarray, masks: np.ndarray, scores: list[float], label_key: int):
        """
        Initialize the MaskSelectionDialog.
        
        :param image: The original image.
        :param masks: List of mask arrays.
        :param scores: List of scores corresponding to each mask.
        :param label_key: The label key to determine the mask color.
        """
        super().__init__()
        self.setWindowTitle('Select Mask')
        
        self.image: np.ndarray = image
        self.masks: np.ndarray = masks
        self.scores: list[float] = scores
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
        
        self.prev_button: QPushButton = QPushButton('Previous')
        self.prev_button.clicked.connect(self.show_previous_mask)
        self.button_layout.addWidget(self.prev_button)
        
        self.next_button: QPushButton = QPushButton('Next')
        self.next_button.clicked.connect(self.show_next_mask)
        self.button_layout.addWidget(self.next_button)
        
        self.select_button: QPushButton = QPushButton('Select')
        self.select_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.select_button)
        
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)
    
    def update_info(self):
        """Update the score and mask count information labels."""
        self.score_label.setText(f"Score: {self.scores[self.current_index]:.3f}")
        self.num_masks_label.setText(f"Masks: {self.current_index + 1} / {len(self.masks)}")
    
    def show_mask(self, index: int):
        """
        Display the mask at the specified index.
        
        :param index: Index of the mask to be displayed.
        """
        mask_image, _ = apply_mask_to_image(self.image, self.masks[index], self.label_key)
        mask_image = self.resize_image(mask_image)
        height, width, _ = mask_image.shape
        bytes_per_line: int = 3 * width
        q_image: QImage = QImage(mask_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        self.image_label.adjustSize()

        self.update_info()
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        label_width: int = self.image_label.width()
        label_height: int = self.image_label.height()

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
    
    def show_previous_mask(self):
        """Show the previous mask in the list."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_mask(self.current_index)
    
    def show_next_mask(self):
        """Show the next mask in the list."""
        if self.current_index < len(self.masks) - 1:
            self.current_index += 1
            self.show_mask(self.current_index)
    
    def get_selected_mask_index(self) -> int:
        """
        Get the index of the currently selected mask.
        
        :return: The index of the selected mask.
        """
        return self.current_index

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        """
        Initialize the ClickableLabel.
        
        :param parent: The parent widget.
        """
        super().__init__(parent)
        self.clicked_position: list[tuple[int, int, int]] = []

    def mousePressEvent(self, event):
        """
        Handle mouse press events. Record the click position and label.

        :param event: The mouse event.
        """
        if event.button() == Qt.LeftButton:
            x: int = event.x()
            y: int = event.y()
            self.clicked_position.append((x, y, 1))
            self.parent().on_image_click(x, y, 1)
            self.update()
        elif event.button() == Qt.RightButton:
            x: int = event.x()
            y: int = event.y()
            self.clicked_position.append((x, y, 0))
            self.parent().on_image_click(x, y, 0)
            self.update()
    
    def paintEvent(self, event):
        """
        Handle paint events. Draw ellipses at the recorded click positions.

        :param event: The paint event.
        """
        super().paintEvent(event)
        painter: QPainter = QPainter(self)
        for x, y, label in self.clicked_position:
            if label == 1:
                pen: QPen = QPen(Qt.red)
                brush: QBrush = QBrush(Qt.red)
            elif label == 0:
                pen: QPen = QPen(Qt.green)
                brush: QBrush = QBrush(Qt.green)

            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.drawText(x, y, f'Label: {label}, (x, y): ({x}, {y})')
        painter.end()

class SAMApp(QWidget):
    def __init__(self):
        """
        Initialize the SAMApp.
        """
        super().__init__()
        self.setWindowTitle('SAM Model UI')
        
        self.image: Optional[np.ndarray] = None
        self.image_list: list[str] = []
        self.predictor: Optional[SamPredictor] = None
        self.input_points: list[list[int]] = []
        self.input_labels: list[int] = []
        self.label_name: Optional[str] = None

        self.sam_checkpoint: Optional[str] = "sam_vit_h_4b8939.pth"
        self.model_type: str = "vit_h"
        self.device: str = "cuda"
        self.model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        self.labels_dict: dict[int, str] = {}
        self.labels_path: str = 'labels.txt'
        self.label_counter: int = 0

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        log_file: str = 'app.log'
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        sys.excepthook = self.handle_exception
        
        self.initUI()
        self.check_and_download_model()
    
    def check_and_download_model(self):
        """
        Check if the model file exists and download if not.
        """
        if not os.path.exists(self.sam_checkpoint):
            reply = QMessageBox.question(self, 'Model Download', 'Model file not found. Do you want to download it?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.logger.info("Model file not found. Downloading...")
                try:
                    response = requests.get(self.model_url, stream=True)
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    progress_dialog = QProgressDialog("Downloading model...", "Cancel", 0, total_size // 1024, self)
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.setMinimumDuration(0)
                    progress_dialog.show()

                    downloaded_size = 0
                    with open(self.sam_checkpoint, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if progress_dialog.wasCanceled():
                                self.logger.info("Model download canceled by user.")
                                os.remove(self.sam_checkpoint)
                                return
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            progress_dialog.setValue(downloaded_size // 1024)
                    
                    self.logger.info("Model downloaded successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to download the model: {e}")
                    QMessageBox.critical(self, 'Download Failed', f'Failed to download the model: {e}')
                    raise
            else:
                self.logger.info("Model download canceled by user.")
                sys.exit()
    
    def handle_exception(self, exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Optional[TracebackType]):
        """
        Handle uncaught exceptions by logging them.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def closeEvent(self, event):
        """
        Handle the close event by logging a message.
        """
        self.logger.info("Close app " + "=" * 40)
        event.accept()
    
    def add_label(self, label_name: str, yolo: bool = False) -> int:
        """
        Added label and return key.
        """
        for key, value in self.labels_dict.items():
            if value == label_name:
                return key

        new_key = self.label_counter
        self.labels_dict[new_key] = label_name
        self.label_counter += 1

        # if yolo:
        #     self.save_labels_dict()
        self.save_labels_dict()

        return new_key
    
    def save_labels_dict(self):
        """
        Save label dict to file.
        """
        with open(os.path.join(self.folder_path, self.labels_path), 'w') as file:
            for key, value in self.labels_dict.items():
                file.write(f"{key}: {value}\n")
    
    def load_labels_dict(self):
        """
        From file loading label dict.
        """
        if os.path.exists(os.path.join(self.folder_path, self.labels_path)):
            with open(os.path.join(self.folder_path, self.labels_path), 'r') as file:
                for line in file:
                    key, value = line.strip().split(': ')
                    self.labels_dict[int(key)] = value
                    self.label_counter = max(self.label_counter, int(key) + 1)
        else:
            try:
                open(os.path.join(self.folder_path, self.labels_path), 'a').close()
                print(f"File '{os.path.join(self.folder_path, self.labels_path)}' created successfully.")
            except Exception as e:
                print(f"Failed to create file '{os.path.join(self.folder_path, self.labels_path)}': {e}")
    
    def update_label_name(self, text: str):
        """
        Update label.
        """
        self.label_name: str = text.strip()
    
    def initUI(self):
        """
        Initialize the user interface.
        """
        self.main_layout: QGridLayout = QGridLayout()

        self.image_label: ClickableLabel = ClickableLabel(self)
        self.image_label.setFixedSize(600, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        placeholder: QImage = QImage(600, 600, QImage.Format_RGB888)
        placeholder.fill(Qt.gray)
        self.image_label.setPixmap(QPixmap.fromImage(placeholder))
        
        self.predicted_label: QLabel = QLabel()
        self.predicted_label.setFixedSize(600, 600)
        self.predicted_label.setAlignment(Qt.AlignCenter)
        
        self.predicted_label.setPixmap(QPixmap.fromImage(placeholder))

        self.predicted_label_mask: QLabel = QLabel()
        self.predicted_label_mask.setFixedSize(600, 600)
        self.predicted_label_mask.setAlignment(Qt.AlignCenter)
        
        self.predicted_label_mask.setPixmap(QPixmap.fromImage(placeholder))
        
        self.main_layout.addWidget(self.image_label, 0, 0)
        self.main_layout.addWidget(self.predicted_label, 0, 1)
        self.main_layout.addWidget(self.predicted_label_mask, 0, 2)
        
        self.side_layout: QVBoxLayout = QVBoxLayout()
        
        self.image_list_widget: QListWidget = QListWidget()
        self.image_list_widget.currentItemChanged.connect(self.display_image)
        self.side_layout.addWidget(self.image_list_widget)
                
        self.load_button: QPushButton = QPushButton('Load Folder')
        self.load_button.clicked.connect(self.load_folder)
        self.side_layout.addWidget(self.load_button)

        self.label_input: QLineEdit = QLineEdit()
        self.label_input.setPlaceholderText('Enter Label Name')
        self.label_input.textChanged.connect(self.update_label_name)
        self.side_layout.addWidget(self.label_input)
        
        self.predict_button: QPushButton = QPushButton('Predict Mask')
        self.predict_button.clicked.connect(self.predict_mask)
        self.side_layout.addWidget(self.predict_button)

        self.save_mask_button: QPushButton = QPushButton('Save Mask')
        self.save_mask_button.clicked.connect(self.save_mask)
        self.side_layout.addWidget(self.save_mask_button)

        # self.save_voc_button = QPushButton('Save YOLO format')
        # self.save_voc_button.clicked.connect(self.save_yolo_labels)
        # self.side_layout.addWidget(self.save_voc_button)

        self.clear_points_button: QPushButton = QPushButton('Clear Points')
        self.clear_points_button.clicked.connect(self.clear_points)
        self.side_layout.addWidget(self.clear_points_button)
        
        self.main_layout.addLayout(self.side_layout, 1, 0, 1, 2)
        
        self.setLayout(self.main_layout)     

        self.logger.info("UI Initialized.")           
        self.initialize_predictor()
    
    def load_folder(self):
        """
        Load images from a folder.
        """
        options: QFileDialog.Options = QFileDialog.Options()
        self.folder_path: str = QFileDialog.getExistingDirectory(self, 'Open Image Folder', options=options)
        if self.folder_path:
            self.image_list = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', '.bmp'))]
            self.image_list_widget.clear()
            for image_path in self.image_list:
                item: QListWidgetItem = QListWidgetItem(os.path.basename(image_path))
                item.setData(Qt.UserRole, image_path)
                self.image_list_widget.addItem(item)
            
            self.logger.info(f"Loaded folder: {self.folder_path} with {len(self.image_list)} images.")
        
        self.load_labels_dict()
    
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

                resized_image: np.ndarray = self.resize_image(self.image)

                height, width, _ = resized_image.shape
                bytes_per_line: int = 3 * width
                q_image: QImage = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image))
                        
                self.predictor: SamPredictor = SamPredictor(self.sam)
                self.predictor.set_image(self.image)

                self.input_points = []
                self.input_labels = []
                self.image_label.clicked_position = []
                self.image_label.update()
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to fit the QLabel dimensions.
        
        :param image: The image to resize.
        :return: The resized image.
        """
        label_width: int = self.image_label.width()
        label_height: int = self.image_label.height()

        image_height, image_width, _ = image.shape
        image_aspect_ratio: float = image_width / image_height
        label_aspect_ratio: float = label_width / label_height

        if image_aspect_ratio > label_aspect_ratio:
            new_width:int = label_width
            new_height: int = int(new_width / image_aspect_ratio)
        else:
            new_height: int = label_height
            new_width: int = int(new_height * image_aspect_ratio)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def initialize_predictor(self):
        """
        Initialize the SAM predictor model.
        """
        try:
            self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            self.sam.to(device=self.device)
            self.logger.info("Model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
    
    def on_image_click(self, x: int, y: int, label: int):
        """
        Handle image click events.
        
        :param x: The x-coordinate of the click.
        :param y: The y-coordinate of the click.
        :param label: The label associated with the click (1 for left click, 0 for right click).
        """
        if self.image is not None:
            label_width: int = self.image_label.width()
            label_height: int = self.image_label.height()

            image_height, image_width, _ = self.image.shape

            scale_x: float = image_width / label_width
            scale_y: float = image_height / label_height

            point: list[int] = [int(x * scale_x), int(y * scale_y)]
            self.input_points.append(point)
            self.input_labels.append(label)
            # print(f"Selected point: {point} with label: {label}")
    
    def predict_mask(self):
        """
        Predict the mask for the selected points.
        """
        if self.image is not None and self.input_points:
            input_points_np: np.ndarray = np.array(self.input_points)
            input_labels_np: np.ndarray = np.array(self.input_labels)
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points_np,
                point_labels=input_labels_np,
                multimask_output=True,
            )
            
            if self.label_name:
                label_key: int = self.add_label(self.label_name)
            else:
                reply = QMessageBox.question(self, 'Enter label name', 'You did not enter a label name',
                                         QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    return
            
            self.show_mask_selection_dialog(masks, scores, label_key)

            self.logger.info(f"Predicted mask for image: {self.image_path}")
    
    def clear_points(self):
        """
        Clear the selected points.
        """
        self.input_points = []
        self.input_labels = []
        self.image_label.clicked_position = []
        self.image_label.update()
        self.logger.info(f"Clear points")
    
    def show_mask_selection_dialog(self, masks: np.ndarray, scores: np.ndarray, label_key: int):
        """
        Show the dialog to select a mask.
        
        :param masks: The predicted masks.
        :param scores: The scores of the masks.
        :param label_key: The label key to determine the mask color.
        """
        dialog = MaskSelectionDialog(self.image, masks, scores, label_key)
        if dialog.exec_() == QDialog.Accepted:
            selected_mask_index: int = dialog.get_selected_mask_index()
            self.show_selected_mask(masks[selected_mask_index], label_key)
    
    def show_selected_mask(self, mask: np.ndarray, label_key: int):
        """
        Show the selected mask.
        
        :param mask: The selected mask.
        :param label_key: The label key to determine the mask color.
        """
        self.mask_image, self.label_image = apply_mask_to_image(self.image, mask, label_key)

        mask_resize_image: np.ndarray = self.resize_image(self.mask_image)
        label_resize_image: np.ndarray = self.resize_image(self.label_image)

        height, width, _ = mask_resize_image.shape
        bytes_per_line: int = 3 * width
        
        q_image: QImage = QImage(mask_resize_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.predicted_label.setPixmap(QPixmap.fromImage(q_image))
        q_image: QImage = QImage(label_resize_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.predicted_label_mask.setPixmap(QPixmap.fromImage(q_image))

    def save_mask(self):
        """
        Save the predicted mask to a file.
        """
        mask_folder_path: str = os.path.join(os.path.dirname(self.folder_path), 'masks')
        if not os.path.exists(mask_folder_path):
            os.makedirs(mask_folder_path)
        
        image_path: str = self.image_path.split('\\')[-1]
        image_path = image_path.split('.')[0]
        image_name_without_extension: str = os.path.splitext(image_path)[0]
        image_name_without_extension = image_name_without_extension.rsplit('.', 1)[0]
        status: bool = cv2.imwrite(f"{mask_folder_path}/{image_name_without_extension}_mask.png", cv2.cvtColor(self.label_image, cv2.COLOR_RGB2BGR))
        if status:
            self.logger.info(f"Mask saved: {mask_folder_path}/{image_name_without_extension}_mask.png")
        else:
            self.logger.error(f"Failed to save mask: {mask_folder_path}/{image_name_without_extension}_mask.png")
    
    def save_yolo_labels(self):
        if self.image is None or not self.input_points or not self.label_name:
            print("Cannot save YOLO labels: No image loaded, points selected, or label name provided.")
            return
        
        labels_folder_path = os.path.join(os.path.dirname(self.folder_path), 'YOLO_labels')
        if not os.path.exists(labels_folder_path):
            os.makedirs(labels_folder_path)
        
        image_path = self.image_path.split('\\')[-1]
        image_path = image_path.split('.')[0]
        image_name_without_extension = os.path.splitext(image_path)[0]
        image_name_without_extension = image_name_without_extension.rsplit('.', 1)[0]
        
        txt_file_path = os.path.join(labels_folder_path, f"{image_name_without_extension}.txt")
        self.write_yolo_labels(txt_file_path)
        print(f"YOLO labels saved: {txt_file_path}")

    def write_yolo_labels(self, txt_file):
        with open(txt_file, 'w') as f:
            for idx, (point, label) in enumerate(zip(self.input_points, self.input_labels)):
                x, y = point
                x_center = x + self.image.shape[1] // 2
                y_center = y + self.image.shape[0] // 2
                width = self.image.shape[1]
                height = self.image.shape[0]
                # YOLO format: <object-class> <x_center> <y_center> <width> <height>
                line = f"{label} {x_center / width} {y_center / height} {width} {height}\n"
                f.write(line)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SAMApp()
    ex.show()
    sys.exit(app.exec_())
