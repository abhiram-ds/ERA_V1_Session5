# Code base for Hand-written digit detection on MNIST using Pytorch
Session 5 Assignment of ERA V1

This repo is the internal engine for our MNIST dataset detecttion using Pytorch. Lets take a look at the components:

## Model

Here you can find code structure for the model class that we have used. Following is the list, for now:

* [Custom Model](model.py)

## Utils

Home for code related to processing and augmentation of images. Current residents:

* [Utils](utils.py)  
  
    Contains following processes:  
  * Image Transformation
  * Downloading the Dataset
  * Dataloaders
  * Display sample images from the dataset
  * Plot loss and accuracy graphs
  * Train and Test functions

## Main Notebook

Home for code related to the core functionality with model creation, training, tuning and testing. Current residents:

* [Session 5 Jupyter Notebook](Session_5_ERA_V1.ipynb)  
  
You can find all the major orchestration code here. Currently available:  

* Train/Test Data Download, Transforms and Loading
* Model Creation and Summary
* Hyperparameter tuning: Optimizer, Learning Rate Scheduler, Loss Criterion
* Plotting the Train and Test Accuracies and Results

### Contributors
Abhiram Gurijala
