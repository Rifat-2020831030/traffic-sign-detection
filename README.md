# Bangla Road Sign Detection using YOLO

[![Roboflow Notebooks](https://media.roboflow.com/notebooks/template/bannertest2-2.png)](https://github.com/roboflow/notebooks)

## Overview

This repository contains a Jupyter Notebook for training a YOLO-based custom object detection model to recognize Bangla road signs. The model is trained using Ultralytics YOLO and Roboflow, leveraging advanced deep learning techniques for accurate detection.
In many practical use cases, like autonomous driving, we need to detect road traffic signs to make logical decisions as a step in performing a task. However, manual road sign detection is impractical. To resolve this issue, we have trained a machine learning model that can detect the most common 9 types of traffic signs with 95.2% mAP score on the test dataset. We used around 4000 manually labeled images as sample data to train a YOLOv11 model. The best output version of the model is then utilized as an inference to run sample images. In our training data, we have particularly selected road signs that are most commonly seen in Bangladeshi roads and have text directions written in Bengali language which makes the problem unique and challenging.

## Methodology
# 2.1 Dataset and Preprocessing
Source: An open-source dataset was acquired from Roboflow, containing approximately 4,000 annotated images of traffic signs.
Image Specifications: All images were standardized to a resolution of 640x640 pixels and stored in JPG format.
Class Distribution: The dataset contained a mixed distribution of instances across classes, reflecting real-world frequency variations.
Data Split: The dataset was partitioned into training, validation, and testing sets with a ratio of 84% / 9% / 8% respectively.
# 2.2 Data Augmentation
To enhance model generalization and robustness against real-world variations, a series of data augmentation techniques were applied during training:
Outputs per training example: 2
Grayscale: Applied to 21% of images.
Brightness: Adjusted between -24% and +0%.
Blur: Applied up to 1.3 pixels.
Noise: Added to up to 1.01% of pixels.
# 2.3 Model Architecture and Training
Model Selection: The YOLOv11 (You Only Look Once v11) architecture was chosen for its state-of-the-art balance of speed and accuracy in object detection tasks.
Training Process: The model was trained for 50 epochs using the default YOLOv11 optimizer and loss functions (Box, Classification, and Distribution Focal Loss). The training process was monitored to ensure proper convergence without significant overfitting.

## Evaluation Metrics
Model performance was quantitatively assessed using standard object detection metrics:
* Precision (P): The accuracy of positive predictions. P = TP / (TP + FP)
* Recall (R): The ability of the model to find all relevant instances. R = TP / (TP + FN)
* mAP@50: Mean Average Precision calculated at an Intersection over Union (IoU) threshold of 0.50. This metric rewards correct class prediction and good localization.
* mAP@50-95: The average mAP over IoU thresholds from 0.50 to 0.95, providing a more stringent measure of localization accuracy.

## Results and Analysis
Training Graph: 
<img width="1434" height="704" alt="Training Graph" src="https://github.com/user-attachments/assets/83203a52-862c-4d9a-baa9-20e1f5ffe3b0" />
Confusion Matrix:
<img width="704" height="681" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/e5442e98-f693-4bc0-97af-4ea303810493" />
Result:
<img width="685" height="685" alt="Result" src="https://github.com/user-attachments/assets/d4cba6fe-ea34-41be-ab54-d3cddda0ee25" />


## Features

- Uses **YOLOv11** for object detection
- Supports GPU acceleration for faster training
- Integrates with **Roboflow** for dataset management
- Optimized training techniques for higher accuracy
- Step-by-step setup guide included

## Installation

Before running the notebook, install the required dependencies:

```bash
pip install "ultralytics<=8.3.40" supervision roboflow
```

Additionally, ensure you have access to a GPU by running:

```bash
!nvidia-smi
```

## Usage

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook bangla_road_sign_detection.ipynb
   ```
3. Follow the instructions in the notebook to train and evaluate the model.

## Dataset

The dataset is managed using **Roboflow**. Ensure you have an API key and follow the notebook instructions to load the dataset.

## Training YOLO Model

- The notebook provides a detailed guide on setting up and training **YOLOv11**.
- You can customize training parameters like epochs, batch size, and image size for optimal performance.
- Currently, the model is trained using 50 epochs and 640*640 image size

## Results & Evaluation

- The trained model's performance is evaluated using **mAP (mean Average Precision)**.
- Predictions can be visualized within the notebook.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

