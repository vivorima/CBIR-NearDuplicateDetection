# CBIR-NearDuplicateDetection Using ResNet Model on a Historical dataset

CBIR-NearDuplicateDetection: A project implementing ResNet-based Content-Based Image Retrieval for identifying near-duplicates in a historical dataset that contains images of war extracted from old newspapers.

## Table of Contents
1. [Introduction](#introduction)
2. [Main Objectives](#main-objectives)
3. [Methodology](#methodology)
4. [Data Preparation](#data-preparation)
5. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
6. [Results](#results)
7. [Technologies Used](#technologies-used)
8. [Usage Instructions](#usage-instructions)
9. [Detailed Training Plots](#detailed-training-plots)


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

The Dataset is a property of the university; ©EYCON dataset.

## Model Training and Fine-Tuning
Fine-tuning a pre-trained ResNet model. 

### Learning parameters
- Learning rate: lr=0.001
- Optimizer: Adam
- Loss function: Cross-Entropy Loss
- Number of epochs: 130, in 3 steps (30 then 50, then 50)
- Batch Size: 32

This is an overview of the plots of our training (This shows the last 50 epochs of training):

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

## Usage Instructions
There are 3 NoteBooks in this repository, each of them represnts a step of our methodology:

- In `Near_Duplicates_Pre_trained_Resnet_Only.ipynb`: We use the pretrained ResNet-152 model on ImageNet to extract features from our images then with a cosine similarity we get the most similar images.
- In `Fine_tuning_Resnet_for_classification.ipynb` : Here we Fine-tune the ResNet-152 model on our dataset (the EYECON dataset) on an image classification task.
- In `Near_Duplicates_using_best_FineTuned_Model.ipynb`: Finally, we use our finetuned ResNet-152 model to extract relevant features from our images then with a cosine similarity we get the most similar images. This is the CBIR (Content-Based Image Retrieval) system.


## Detailed Training Plots

In this section, we discuss the performance of our ResNet model during the training phase over the last 50 epochs as visualized in the following performance metrics plots:
![All Metrics](https://github.com/vivorima/CBIR-NearDuplicateDetection/blob/de86a930f49654bad6177f0b2121b65f0803e585/all%20plots.png "All training Metrics")


- ### Loss over Epochs: shows the model's training and validation loss.
The training loss decreases over time, which indicates that the model is learning and improving its predictions on the training data. The validation loss, while generally following the training loss, shows some fluctuations (the peaks), suggesting moments of learning and overfitting, I believe this is due to the class imbalance and the limited batch size.

- ### Precision over Epochs: tracks the precision metric for both training and validation sets.
Precision reflects the model's accuracy in classifying an image as belonging to a relevant class (scene). The plot exhibits a general upward suggesting that the model's ability to correctly identify relevant images but it fluctuates across epochs.

- ### Recall over epochs : represents the recall metric for both training and validation sets.
This metric indicates the model's ability to find all relevant instances in the dataset. Both training and validation recall show a volatile pattern, indicating inconsistency in the model's sensitivity to detecting all relevant samples through the epochs.

- ### F1 Score over Epochs : shows the F1 score for the training and validation sets.
The F1 score is the mean of precision and recall, providing a single metric that balances both concerns. The plot indicates that model's balance between precision and recall is not yet stable, indicating the need for more training, more data and specially more balance.

Overall, these plots suggest that while the model is learning and improving, there is room for improvement in terms of generalization and stability of the performance metrics across epochs. 

