# SAM Model UI Interface - Enhancing Efficiency in Semantic Segmentation Labeling

This project utilizes the SAM (Segment-Anything Model) from Facebook Research's [segment-anything](https://github.com/facebookresearch/segment-anything) repository. It aims to provide an intuitive and efficient user interface (UI) to assist users in generating and editing semantic segmentation labels.

## Project Background

Semantic segmentation is a crucial task in computer vision, aiming to assign semantic category labels to each pixel in an image, such as humans, vehicles, roads, etc. However, manual annotation of large-scale datasets is time-consuming and tedious. This project leverages the powerful SAM model and combines it with an intuitive UI to offer a more efficient way to create and edit semantic segmentation labels.

## Key Features

- **Semantic Segmentation with SAM Model**
  - Utilize the pretrained SAM model for image semantic segmentation, generating initial label outputs.
  
- **Interactive UI**
  - Provide a user-friendly interface that allows users to edit and refine generated labels to ensure accuracy and completeness.
  
- **Real-time Preview and Saving**
  - Instantly preview the effects of edited labels and support saving edited results in standard formats such as PNG, JSON or XML.

## How to Use

### Environment Setup

```bash
pip install -r requirements.txt 
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```
### Running the UI

```bash
python ui_main.py
```
<p float="left">
    <img src="assets/UI_interface1.png?raw=true" width="49.7%" />
    <img src="assets/UI_interface2.png?raw=true" width="49.7%" />
    <img src="assets/UI_interface3.png?raw=true" width="49.7%" />
    <img src="assets/UI_interface4.png?raw=true" width="49.7%" />
</p>

### User Interface Buttons

In this project, QPushButton widgets are utilized for various functionalities in the graphical user interface (GUI). Each button is connected to a corresponding method to handle user interactions effectively:

- **Load Folder Button**: This button (`Load Folder`) is used to trigger the loading of a folder or directory.

- **Predict Mask Button**: The `Predict Mask` button initiates the process of predicting a mask or segmentation for the loaded data.

- **Save Mask Button**: Clicking the `Save Mask` button saves the generated mask or segmentation results to a file. Handle the saving process and ensure the output is stored in a specified format (e.g., PNG, JSON).

- **Clear Points Button**: The `Clear Points` button allows users to reset or clear any marked points or annotations on the interface.

## License
The model is licensed under the [Apache 2.0 license](LICENSE).