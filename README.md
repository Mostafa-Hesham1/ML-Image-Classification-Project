# Image Classification Project

This repository contains code for an image classification project using TensorFlow/Keras with various deep learning models (CNN, VGG16, ResNet50, InceptionV3, MobileNetV2) on the Intel Image Classification dataset.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

## Overview <a name="overview"></a>

This repository contains code to train, evaluate, and use several deep learning models for classifying images into six categories: buildings, forest, glacier, mountain, sea, and street. It includes model definitions, training scripts, and an interactive classification interface using Gradio.

## Installation <a name="installation"></a>

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/image-classification-project.git
   cd image-classification-project
Dataset <a name="dataset"></a>
The project uses the Intel Image Classification dataset available on Kaggle.

Download the dataset from Kaggle:
kaggle datasets download -d puneet6060/intel-image-classification
unzip intel-image-classification.zip
rm intel-image-classification.zip
##Organize the dataset as follows:
/content
├── seg_train
│   ├── buildings
│   ├── forest
│   ├── glacier
│   ├── mountain
│   ├── sea
│   └── street
└── seg_test
    ├── buildings
    ├── forest
    ├── glacier
    ├── mountain
    ├── sea
    └── street
Models <a name="models"></a>
1. Custom CNN Model
Architecture: Custom defined CNN model with convolutional layers, max pooling, dropout, and dense layers.
File: cnn_model.py
Training: Trained with ImageDataGenerator for data augmentation.
2. VGG16 Model
Architecture: Transfer learning using VGG16 pre-trained on ImageNet.
File: vgg16_model.py
Training: Fine-tuned with additional dense layers.
3. ResNet50 Model
Architecture: Transfer learning using ResNet50 pre-trained on ImageNet.
File: resnet50_model.py
Training: Global Average Pooling and dense layers added for classification.
4. InceptionV3 Model
Architecture: Transfer learning using InceptionV3 pre-trained on ImageNet.
File: inceptionv3_model.py
Training: Includes fine-tuning of top layers.
5. MobileNetV2 Model
Architecture: Transfer learning using MobileNetV2 pre-trained on ImageNet.
File: mobilenetv2_model.py
Training: Fine-tuned for image classification tasks.
Training <a name="training"></a>
Each model is trained using ImageDataGenerator for data augmentation.
Training scripts are provided for each model in their respective files (*_model.py).
Evaluation <a name="evaluation"></a>
Evaluation metrics such as Loss, Accuracy, Precision, Recall, and F1 Score are computed for each model.
A summary of evaluation results is provided in the README.
Usage <a name="usage"></a>
Interactive Classification Interface
Use Gradio to interactively classify images using trained models.

##Run the interface:
python app.py

Acknowledgements <a name="acknowledgements"></a>
Intel for providing the dataset.
TensorFlow and Keras communities for the deep learning frameworks.
Gradio for the interactive interface.


