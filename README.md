# Face-Mask-Detector 

## Members :
 * Pulkit Dhingra
 * Prachi Porwal
 * Purvika Panwar

## Pipeline
2 steps to be performed are - 

**Training**: Loading dataset from disk, training a model on this dataset, saving face mask detector to disk.
**Deployment**: Loading mask detector, performing face detection, categorizing each as with_mask or without_mask.

## About files (directory)

  * Dataset: 
  Contains 690 images under with_mask category and 689 images under without_mask category.
  * Examples: 
  Contains sample images to be tested 
  * face_detector: 
  
  * train_mask_detector.py: 
  Deep Learning Model 
  * detect_mask_image.py: 
  
  * detect_mask_video.py: 
  
  * mask_detector.model: 
  
  * plot.png: 
  
## Functions/Libraries/Classes and their use

### Used in train_mask_detector.py file
**argparse**: Creates a simple CLI in python.

**load_image**: Loads the image from the path.

**img_to_array**: Converts a PIL image instance to np array, returns a 3D array (height, widht, channel) / channel can be placed at first axis or last.

**preprocess_input**: Adequate your image to the format the model requires, creates a batch of image (samples, height, width, channel), scale the pixel intensities.

**LabelBinarizer.fit_transform**: Performs one hot encoding (mapping of categorical data into numerical data) 

**to_categorical**: Integer Encoding -> Binary vector (matrix)

**train_test_split**: split the entire dataset(matrices) into 2 parts (train and test) 

**ImageDataGenerator**: Accepts a batch of images and apply a series of random transformations to each image in batch. Replaces original batch with new. Performs Data Augmentation.

**MobileNetV2**: Efficient CNN for computer vision applications.

**AveragePooling2D**: calculating the average for each patch of the feature map.

**Flatten**: Pooled feature map -> vector of input data

**Dense**: Regular layer that performs activation.

**Dropout**: This layer tackles overfitting. Param takes fraction of neurons to drop.

**argmax**: indices of max value along the axis

**H.history**: It holds the record of loss values and metrix values.

### Used in detect_mask_video.py file

## Some terminology

**Confusion Metrics**: Actual class on left, predicted class on top

**Accuracy**: Ratio of correctly predicted observation to the total observations. (TP+TN/ TP+TN+FP+FN)

**Precision**: Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. (TP/ TP+FP)

**Recall**: Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. (TP / TP + FN). Recall above 0.5 is good.

**F1 score**: Weighted average of Precision and Recall. 2 * (P * R) / (R + P).

**Support**: No. of occurrences of each class in Y_true here testY.

## Trends in graph

Validation loss curve is lower than training loss curve. 
Reason1 : Regularization applied during training (droupout layer) and not during validation.
Reason2 : Validation set may be easier than the training set.

## How to run:

First and foremost is to train the model, which is already done in this case.
```
python train_mask_detector.py --dataset dataset
```

For testing model against sample images
```
python detect_mask_image.py --image examples/example_01.png
```

Finally for live video testing
```
python detect_mask_video.py
```
