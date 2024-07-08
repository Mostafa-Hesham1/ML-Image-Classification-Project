# Image Classification Project

## Overview

This repository contains code for an image classification project using TensorFlow/Keras with various deep learning models (CNN, VGG16, ResNet50, InceptionV3, MobileNetV2) on the Intel Image Classification dataset.



## Download the dataset from Kaggle
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip
rm intel-image-classification.zip
## Organize the dataset
## /content
### ├── seg_train
#### │   ├── buildings
#### │   ├── forest
#### │   ├── glacier
#### │   ├── mountain
#### │   ├── sea
#### │   └── street
### └── seg_test
####     ├── buildings
####     ├── forest
####     ├── glacier
####     ├── mountain
 ####    ├── sea
  ####    └── street
## Models 

Each model is trained to classify images into one of the following categories: **buildings**, **forest**, **glacier**, **mountain**, **sea**, or **street**. Here's a brief overview of each model's architecture and its predictive capabilities:

- **Custom CNN**: A custom convolutional neural network designed specifically for this project, capable of learning spatial hierarchies in images and predicting the correct category.

- **VGG16**: Utilizes transfer learning from a pre-trained VGG16 model on ImageNet, leveraging its deep architecture to recognize intricate patterns in images for accurate classification.

- **ResNet50**: Another transfer learning approach using ResNet50, known for its residual blocks that help in training deeper networks effectively, enhancing accuracy in image classification tasks.

- **InceptionV3**: Employs the InceptionV3 architecture, which uses multiple filters at each layer to capture complex features in images, thus improving the model's ability to distinguish between different categories.

- **MobileNetV2**: Optimized for mobile and embedded vision applications, MobileNetV2 provides a lightweight yet powerful solution for image classification tasks, balancing between accuracy and computational efficiency.
  

## Training
- Each model is trained using ImageDataGenerator for data augmentation.
- Training scripts are provided for each model in their respective files (*_model.py).
## Evaluation
- Evaluation metrics such as Loss, Accuracy, Precision, Recall, and F1 Score are computed for each model.
### Evaluation Metrics

| Model        | Loss   | Accuracy | Precision | Recall | F1 Score |
|--------------|--------|----------|-----------|--------|----------|
| Custom CNN   | 0.4785 | 83.60%   | 85.72%    | 81.23% | 83.42%   |
| VGG16        | 0.3066 | 88.17%   | 89.74%    | 87.13% | 88.42%   |
| ResNet50     | 0.5779 | 82.67%   | 84.89%    | 81.43% | 83.12%   |
| InceptionV3  | 0.2966 | 89.27%   | 90.46%    | 87.53% | 88.97%   |
| MobileNetV2  | 0.3518 | 87.07%   | 88.56%    | 85.17% | 86.83%   |

## Usage
### Interactive Classification Interface
- Use Gradio to interactively classify images using trained models.
- Run the interface:
## Acknowledgements
- Intel for providing the dataset.
- TensorFlow and Keras communities for the deep learning frameworks.
- Gradio for the interactive interface.

