# Semantic segmentation with MobileNetV3  <!-- omit in TOC -->

This repository contains the code for training of MobileNetV3 for segmentation as well as default model for classification. Every module here is subject for subsequent customizing.

## Content <!-- omit in TOC -->

- [Requirements](#requirements)
- [Quick setup and start](#quick-setup-and-start)
  - [Preparations](#preparations)
  - [Run](#run)
- [CNN architectures](#cnn-architectures)
- [Loss functions](#loss-functions)
- [Augmentations](#augmentations)
- [Training](#training)
- [Convert to TensorFlow Lite](#convert-to-tensorflow-lite)
- [Pretrained models](#pretrained-models)
- [Projects use the MobileNetV3-segm model implementation](#projects-use-the-mobilenetv3-segm-model-implementation)

## Requirements
    Machine with an NVIDIA GPU
    NVIDIA driver >= 418
    CUDA >= 10.1
    Docker >= 19.03
    NVIDIA Container Toolkit (https://github.com/NVIDIA/nvidia-docker)

## Quick setup and start

### Preparations 

* Clone the repo, build a docker image using provided Makefile and Dockerfile. 

    ```
    git clone 
    make build
    ```
* The final folder structure should be:
  
    ```
    Semantic-segmentation-with-MobileNetV3
    ├── data
    ├── notebooks
    ├── modules
    ├── train
    ├── Dockerfile
    ├── Makefile
    ├── requirements.txt
    ├── README.md
    ```

### Run

* The container could be started by a Makefile command. Training and evaluation process was made in Jupyter Notebooks so Jupyter Notebook should be started.

    ```
    make run
    jupyter notebook --allow-root
    ```

## CNN architectures

MobileNetV3 backnone with Lite-RASSP modules were implemented.
Architecture may be found in [modules/keras_models.py](modules/keras_models.py)

## Loss functions

F-beta and FbCombinedLoss (F-beta with Cross Entropy) losses were implemented.
Loss functions may be found in [modules/loss.py](modules/loss.py)

## Augmentations

There were implemented the following augmentations:
Random rotation, random crop, scaling,
horizontal flip, brightness, gamma and contrast augmentations,
Gaussian blur and noise.
  
Details of every augmentation may be found in [modules/segm_transforms.py](modules/segm_transforms.py)
    
## Training
 
Training process is implemented in [notebooks/train_mobilenet.ipynb](notebooks/train_mobilenet.ipynb) notebook.

Provided one has at least PicsArt AI Hackathon dataset and Supervisely Person Dataset it is only needed to run every cell in the notebook subsequently.
 
## Convert to TensorFlow Lite
 
To successfully convert this version of MobileNetV3 model to TFLite optional argument "training" must be removed from every batchnorm layer in the model and after that pretrained weights may be loaded and notebook cells for automatic conversion may be executed.

[notebooks/convert2tflite.ipynb](notebooks/convert2tflite.ipynb) notebook contains model conversion sample scripts with and without quanization.

## Pretrained models

Only person segmentation datasets were used for training models in this project: PicsArt AI Hackathon dataset and Supervisely Person Dataset.

Trained Keras model (input size 224x224 px) may be found [here](https://my.pcloud.com/publink/show?code=XZUDrwkZBrdvwMDebrz5Q97Jue4cxXFgYys7).

Trained model converted to a TensorFlow Lite FlatBuffer may be found [here](https://my.pcloud.com/publink/show?code=XZqrpLkZJixoFPoWXL0PRvHLBIGzKf1ecNKy).

The same model but quantized after training may be downloaded via this [link](https://my.pcloud.com/publink/show?code=XZLcpLkZMIBz7TIOKG7gAwCxqNWGJLfpdsuy).

*Note:* The model was trained with TF2.0, so, it may contain some bugs as compared with the current TF version.

## Projects use the MobileNetV3-segm model implementation

* Real-time CPU person segmentation in video calls: [repo](https://github.com/NikolasEnt/PersonMask_TFLite)
