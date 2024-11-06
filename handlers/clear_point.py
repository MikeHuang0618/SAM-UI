def clear_points(sam_ui):
    """
    Clear the selected points.
    """
    sam_ui.input_points = []
    sam_ui.input_labels = []
    sam_ui.image_label.clicked_position = []
    sam_ui.image_label.update()
    sam_ui.logger.info("Clear points")
