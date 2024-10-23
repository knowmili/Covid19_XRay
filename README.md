# Covid-19 Chest X-Ray Classification

This project aims to detect classify chest X-ray images using a Deep Convolutional Neural Network. The model architecture uses pre-trained weights from VGG19 and fine-tunes it to achieve accurate multi-class classification.

## Dataset

The dataset is sourced from the [Covid19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data). It includes X-ray images for the following classes:

- COVID-19
- Normal
- Lung Opacity
- Viral Pneumonia

## Project Structure

- **notebooks**: Jupyter notebooks used for exploration and initial experiments
- **model**: Model weights and checkpoints

## Model

Used VGG19, a deep convolutional neural network architecture pre-trained on the ImageNet dataset, known for its ability to extract rich feature representations from images. VGG19 consists of 19 layers and has been widely used in image classification tasks due to its depth and simplicity.

**Architecture**:

- 3x Conv2D (ReLU) + MaxPooling
- Flatten layer
- Dense (ReLU) + Dropout
- Output (Softmax) for 4 classes

Download the dataset from [here](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data).

## Results

The model achieved **92.56% accuracy** on the test set. Below is the confusion matrix:

![Prediction Example](/confusion_mat.png)
