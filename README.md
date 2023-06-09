# Food-Classifier
It uses Pytorch and CNN model to identify food items
# Food Image Classification

This repository contains code for a food image classification project. The goal of this project is to build a convolutional neural network (CNN) model that can classify food images into different categories. The code uses the PyTorch library for deep learning and image processing.

## Dataset

The dataset used for training, validation, and testing consists of food images. It is divided into three directories: training, validation, and test. Each image is associated with a label indicating the food category.

The dataset is provided here : https://www.kaggle.com/datasets/trolukovich/food11-image-dataset
Make sure to downnload the dataset and upload it on google drive before running the code.

## Getting Started

To run the code, follow these steps:

1. Install the necessary dependencies. The code requires Python 3.x, PyTorch, and torchvision. You can install them using pip:
    pip install torch torchvision

2. Mount your Google Drive by executing the following code:
   from google.colab import drive
   drive.mount('/content/drive')


3. Import the required libraries and unzip the dataset using the provided code.

4. Set up the necessary configurations and hyperparameters, such as the number of epochs and batch size.

5. Run the code to train the CNN model on the training dataset and evaluate its performance on the validation dataset. The best model will be saved during training.

6. Once the training is complete, the code will load the best model and make predictions on the test dataset. The predicted labels and corresponding images will be displayed.

## Results

The code outputs the training and validation loss, as well as the accuracy, for each epoch during the training process. The best model based on the validation accuracy is saved as `sample_best.ckpt`.

