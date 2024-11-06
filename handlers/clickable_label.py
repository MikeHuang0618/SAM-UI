from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import QLabel


class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clicked_position = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            self.clicked_position.append((x, y, 1))
            if hasattr(self.parent(), "on_image_click"):
                self.parent().on_image_click(x, y, 1)
            self.update()
        elif event.button() == Qt.RightButton:
            x, y = event.x(), event.y()
            self.clicked_position.append((x, y, 0))
            if hasattr(self.parent(), "on_image_click"):
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
            painter.drawText(x, y, f"Label: {label}, (x, y): ({x}, {y})")
        painter.end()


# on_image_click function
def on_image_click(self, x: int, y: int, label: int):
    if self.image is not None:
        label_width, label_height = self.image_label.width(), self.image_label.height()
        image_height, image_width, _ = self.image.shape
        scale_x, scale_y = image_width / label_width, image_height / label_height
        point = [int(x * scale_x), int(y * scale_y)]
        self.input_points.append(point)
        self.input_labels.append(label)
