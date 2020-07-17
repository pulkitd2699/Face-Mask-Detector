# Face-Mask-Detector 

## Members :
 * Pulkit Dhingra
 * Prachi Porwal
 * Purvika Panwar

### Pipeline
2 steps to be performed are - 

**Training**: Loading dataset from disk, training a model on this dataset, saving face mask detector to disk.
**Deployment**: Loading mask detector, performing face detection, categorizing each as with_mask or without_mask.

### About files (directory)

  * Dataset
  Contains 690 images under with_mask category and 689 images under without_mask category.
  * Examples
  Contains sample images to be tested 
  * face_detector
  
  * train_mask_detector.py
  Deep Learning Model 
  * detect_mask_image.py
  
  * detect_mask_video.py 
  
  * mask_detector.model
  
  * plot.png
  
### Functions/Libraries/Classes and their use

#### Used in train_mask_detector.py file
**argparse**: Creates a simple CLI in python.
**load_image**: Loads the image from the path.
**img_to_array**: Converts a PIL image instance to np array, returns a 3D array (height, widht, channel) / channel can be placed at first axis or last.
**preprocess_input**: Adequate your image to the format the model requires, creates a batch of image (samples, height, width, channel), scale the pixel intensities.
**LabelBinarizer.fit_transform**: Performs one hot encoding (mapping of categorical data into numerical data) 
**to_categorical**: Integer Encoding -> Binary vector (matrix)
**train_test_split**: split the entire dataset(matrices) into 2 parts (train and test) 
**ImageDataGenerator**: Accepts a batch of images and apply a series of random transformations to each image in batch. Replaces original batch with new. Performs Data Augmentation.
****


#### Used in detect_mask_image.py file


#### Used in detect_mask_video.py file

### 
