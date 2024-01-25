# CBIR-NearDuplicateDetection
CBIR-NearDuplicateDetection: A project implementing ResNet-based Content-Based Image Retrieval for identifying near-duplicates

# Content-Based Image Search with ResNet

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Technologies Used](#technologies-used)
6. [Data Preparation](#data-preparation)
7. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
8. [Dataset and Model Path Configuration](#dataset-and-model-path-configuration)
9. [Custom Dataset Class](#custom-dataset-class)
10. [Installation and Usage Instructions](#installation-and-usage-instructions)

## Introduction
This project aims to develop a content-based image retrieval system using Convolutional Neural Networks (CNNs), specifically the ResNet-152 model. In collaboration with historical researchers from Université Paris Cité, this project facilitates the retrieval of relevant historical images published in newspapers and journals.

## Objectives
- Develop a deep learning model to create effective visual representations.
- Identify images of the same scene using the CNN model.

## Methodology
- Use of ResNet-152 pre-trained on ImageNet.
- Fine-tuning the ResNet-152 model on a specific dataset for image classification.
- Feature extraction and use in a CBIR (Content-Based Image Retrieval) system.

## Results
The project achieved promising results with good precision and F1 scores, demonstrating the effectiveness of the chosen approach, even with challenges posed by class imbalance and the limited size of the dataset.

## Technologies Used
- Python
- Libraries: PyTorch, Pandas, NumPy, PIL, sklearn, etc.

## Data Preparation
The project requires specific data preparation, including splitting into training, validation, and test sets. The data should be formatted as indicated in the notebook.

## Model Training and Fine-Tuning
This project involves fine-tuning a pre-trained ResNet model. Details on the parameters used and the fine-tuning method are explained in the notebook.

## Dataset and Model Path Configuration
The notebook uses specific paths for the dataset and the model. Clear instructions on how to configure these paths in your local environment are provided.

## Custom Dataset Class
A `SceneDataset` class is implemented to handle the image dataset. A description of its purpose and use is included in the notebook.

## Installation and Usage Instructions

## Images for Plots
- Plot images demonstrating model performance, such as loss and accuracy graphs, confusion matrices, or example retrieval results. 
