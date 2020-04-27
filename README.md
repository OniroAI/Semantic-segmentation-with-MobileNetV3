# Semantic segmentation with MobileNetV3

This repository contains the code for training of MobileNetV3 for segmentation as well as default model for classification. Every module here is subject for subsequent customizing.
## Content
*  [Requirements](#requirements)
*  [Quick setup and start](#quickstart)
*  [CNN architectures](#cnn)
*  [Loss functions](#loss_functions)
*  [Augmentations](#augmentations)
*  [Training](#training)
*  [Trained model](#trained_model)

## Requirements  <a name="requirements"/>
    Machine with an NVIDIA GPU
    NVIDIA driver >= 418
    CUDA >= 10.1
    Docker >= 19.03
    NVIDIA Container Toolkit (https://github.com/NVIDIA/nvidia-docker)
    
## Quick setup and start  <a name="quickstart"/>

### Preparations 

* Clone the repo, build a docker image using provided Makefile and Dockerfile. 

    ```
    git clone 
    make build
    ```
* The final folder structure should be:
  
    ```
    airbus-ship-detection
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

## CNN architectures <a name="cnn"/> 

MobileNetV3 backnone with Lite-RASSP modules were implemented.
Architecture may be found in [modules/keras_models.py](modules/keras_models.py)

## Loss functions  <a name="loss_functions"/>

F-beta and FbCombinedLoss (F-beta with Cross Entropy) losses were implemented.
Loss functions may be found in [modules/loss.py](modules/loss.py)

## Augmentations <a name="augmentations"/>

There were implemented the following augmentations:
Random rotation, random crop, scaling,
 horizontal flip, brightness, gamma and contrast augmentations,
  Gaussian blur and noise.
  
Details of every augmentation may be found in [modules/segm_transforms.py](modules/segm_transforms.py)
    
## Training  <a name="training"/>
 
 Training process is implemented in [notebooks/train_mobilenet.ipynb](notebooks/train_mobilenet.ipynb) notebook.
 
 Provided one has at least Pixart and Supervisely Person Dataset it is only needed to run every cell in the notebook subsequently.
 
## Trained model  <a name="trained_model"/>
 
 To successfully convert this version of MobileNetV3 model to TFLite optional argument "training" must be removed from every batchnorm layer in the model and after that pretrained weights may be loaded and notebook cells for automatic conversion may be executed.

 Only person segmentation datasets were used for training models in this project: Pixart and Supervisely Person Dataset.

 Trained Keras model may be found [here](https://my.pcloud.com/publink/show?code=XZUDrwkZBrdvwMDebrz5Q97Jue4cxXFgYys7).
 
 Trained model converted to a TensorFlow Lite FlatBuffer may be found [here](https://my.pcloud.com/publink/show?code=XZqrpLkZJixoFPoWXL0PRvHLBIGzKf1ecNKy).
 
 The same model but quantized after training may be downloaded via this [link](https://my.pcloud.com/publink/show?code=XZLcpLkZMIBz7TIOKG7gAwCxqNWGJLfpdsuy).