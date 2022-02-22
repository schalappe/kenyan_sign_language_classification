# kenyan_sign_language_classification

## Description
A common misconception of sign language is that it is the same everywhere; in reality there are as many as 300 different 
languages (approximately 50 of these from Africa) with new signs evolving each day as a need appears. 
Kenyan Sign Language (KSL) is used in Kenya and Somalia, and there are different dialects depending on what region you 
are in. It is used by over half of Kenya's estimated 600 000-strong deaf population.

The objective is to build a model to recognise ten different everyday KSL signs present in the images, using machine 
learning or deep learning algorithms.

Details about the dataset and challenge can be seen on [challenge page](https://zindi.africa/competitions/kenyan-sign-language-classification-challenge)

## Dataset
The data was collected by 800 taskers from Kenya, Mexico and India. There are nine classes, each a different sign.

The dataset comprises of the following files:

- [Images.zip (~1.1GB)](https://api.zindi.africa/v1/competitions/kenyan-sign-language-classification-challenge/files/Images.zip): is a zip file that contains all images in test and train.
- [Train.csv](https://api.zindi.africa/v1/competitions/kenyan-sign-language-classification-challenge/files/Train.csv): contains the target. This is the dataset that you will use to train your model.


## Pre-processing
Images in the dataset did not have fixed size therefore it was mandatory to resize them for training.

The size of dataset is small, so we needed to add more data for training, and we used data augmentations. 
For Data Augmentation we performed:

- Random saturation
- Random HUE
- Random brightness
- Random contrast
- Gaussian noise
- Randomly extract a part of image

## Model Architectures

The following architecturs are included in the library of models:
* ResNet
* DenseNet
* EfficientNet
* NasNet

## Common configuration
The following configurations is applied on all trails in hyperparameter optimization process:
* [Gradient Centralization](https://arxiv.org/abs/2004.01461)
