# SAM Model UI Interface - Enhancing Efficiency in Semantic Segmentation Labeling

This project utilizes the SAM2 (Segment-Anything Model 2) from Facebook Research's [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) repository. It aims to provide an intuitive and efficient user interface (UI) to assist users in generating and editing semantic segmentation labels.

## New update
Segment-Anything Model update to Segment-Anything Model 2.

### Notice
The Windows version of PyTorch doesn't build Flash Attention. If you really want to use it on Windows, you will need to build Flash Attention yourself, as it is only available in the Linux version of PyTorch. Therefore, I recommend running this project on Ubuntu. The official guidance suggests using Ubuntu under WSL2 for the best results.

## Project Background

Semantic segmentation is a crucial task in computer vision, aiming to assign semantic category labels to each pixel in an image, such as humans, vehicles, roads, etc. However, manual annotation of large-scale datasets is time-consuming and tedious. This project leverages the powerful `SAM2` model and combines it with an intuitive UI to offer a more efficient way to create and edit semantic segmentation labels.

## Key Features

- **Semantic Segmentation with SAM2 Model**
  - Utilize the pretrained `SAM2` model for generating initial label outputs from images.
  
- **Interactive UI**
  - A user-friendly interface that allows for the editing and refinement of generated labels to ensure accuracy and completeness.
  
- **Real-time Preview and Saving**
  - Instant preview of edited labels and support for saving results in standard formats such as PNG, JSON, or XML.

## How to Use

### Environment Setup

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```
Need install segment-anything model 2.
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```

### Running the UI

```bash
python ui_main.py
```
<p float="left">
    <img src="assets/UI_interface1.jpg?raw=true" width="49.7%" />
    <img src="assets/UI_interface2.jpg?raw=true" width="49.7%" />
    <img src="assets/UI_interface3.jpg?raw=true" width="49.7%" />
</p>

### User Interface Buttons

The UI employs QPushButton widgets for various functionalities. Each button is connected to a method to handle user interactions:

- **Load Folder Button**: Initiates the process of loading a folder or directory.

- **Predict Mask Button**: Begins the prediction of a mask or segmentation for the loaded data.

- **Save Mask Button**: Saves the generated mask or segmentation results to a file, ensuring the output is in the specified format (e.g., PNG, JSON).

- **Clear Points Button**: Resets or clears any marked points or annotations on the interface.

## TODO
&#9744; Save the coordinates in YOLO-seg format.  
&#9744; Added the coco and Pascal VOC format.

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Feedback and Issues
We welcome feedback and issues from users. If you encounter any problems or have suggestions for improvements, please [open an issue](https://github.com/MikeHuang0618/SAM-UI/issues/new) on our GitHub repository.
