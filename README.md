# CBIR-NearDuplicateDetection Using ResNet Model on a Historical dataset

CBIR-NearDuplicateDetection: A project implementing ResNet-based Content-Based Image Retrieval for identifying near-duplicates in a historical dataset that contains images of war extracted from old newspapers.

## Table of Contents
1. [Introduction](#introduction)
2. [Main Objectives](#main-objectives)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Technologies Used](#technologies-used)
6. [Data Preparation](#data-preparation)
7. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
8. [Dataset and Model Path Configuration](#dataset-and-model-path-configuration)
9. [Custom Dataset Class](#custom-dataset-class)
10. [Installation and Usage Instructions](#installation-and-usage-instructions)

## Introduction

This is a Research project in collaboration with historian researchers affiliated with Université Paris Cité.
The aim of this project is to enable historical researchers to find all the images taken during historical events and published in newspapers and magazines.
Technically, this project aims to develop a content-based image retrieval system using a Convolutional Neural Network, specifically the ResNet-152 model.

## Main Objectives
- Develop a deep learning model to extract effective visual features.
- Identify images of the same scene using the CNN model.

## Methodology (Experiments)
- Use of ResNet-152 pre-trained on ImageNet.
- Fine-tuning the ResNet-152 model on our dataset (the EYECON dataset) for an image classification task.
- Feature extraction and use in a CBIR (Content-Based Image Retrieval) system.

## Data Preparation
The project requires specific data preparation, including splitting into training, validation, and test sets. 
The dataset contains 71 different classes, each representing a "scene". In total, the dataset contains 268 images. 
This is how we split them:
- Training set: 160 images
- Validation set: around 50 images.
- Test set: around 50 images. (Note: There seems to be a typo in the original text, mentioning 'Validation set' twice.)

This Graph showcases the class imbalance in the dataset:
![Class imbalance in the dataset](https://github.com/vivorima/CBIR-NearDuplicateDetection/blob/f6654bb32e89c951161acc13837ddea2ae179ec4/Pr%C3%A9sentation%20projet%20resent.png "Class imbalance in the dataset")

## Model Training and Fine-Tuning
Fine-tuning a pre-trained ResNet model. 

### Learning parameters
- Learning rate: lr=0.001
- Optimizer: Adam
- Loss function: Cross-Entropy Loss
- Number of epochs: 130, in 3 steps (30 then 50, then 50)
- Batch Size: 32

This is an overview of the plots of our training:

![Loss and Accuracy Metrics](https://github.com/vivorima/CBIR-NearDuplicateDetection/blob/f6654bb32e89c951161acc13837ddea2ae179ec4/overview.png "Loss and Accuracy Metrics")

## Results of our evaluation
The model achieved promising results with good precision and F1 scores, demonstrating the effectiveness of the chosen approach, even with challenges posed by class imbalance and the limited size of the dataset. NDCG measures the relevance with which the system ranks the most similar images at the top of the list.

| Metric     | Value |
|------------|-------|
| Precision  | 90%   |
| Recall     | 85%   |
| F1 score   | 85%   |
| nDCG Score | 46%   |

The nDCG score of 0.463 indicates room for improvement in image relevance ranking, while precision of 0.905 and recall of 0.852 show the model's strong discrimination capability and sensitivity. The F1 score of 0.854 reveals a good balance between precision and recall.

## Technologies Used
- Python
- Libraries: PyTorch, Pandas, NumPy, PIL, sklearn, etc.
