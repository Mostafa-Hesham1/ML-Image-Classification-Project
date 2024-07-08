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
1. Custom CNN Model

- Architecture: Custom defined CNN model with convolutional layers, max pooling, dropout, and dense layers.
- File: cnn_model.py
- Training: Trained with ImageDataGenerator for data augmentation.
  
2. VGG16 Model

- Architecture: Transfer learning using VGG16 pre-trained on ImageNet.
- File: vgg16_model.py
- Training: Fine-tuned with additional dense layers.
  
3. ResNet50 Model

- Architecture: Transfer learning using ResNet50 pre-trained on ImageNet.
- File: resnet50_model.py
- Training: Global Average Pooling and dense layers added for classification.
  
4. InceptionV3 Model

- Architecture: Transfer learning using InceptionV3 pre-trained on ImageNet.
- File: inceptionv3_model.py
- Training: Includes fine-tuning of top layers.

5. MobileNetV2 Model

- Architecture: Transfer learning using MobileNetV2 pre-trained on ImageNet.
- File: mobilenetv2_model.py
- Training: Fine-tuned for image classification tasks.
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

