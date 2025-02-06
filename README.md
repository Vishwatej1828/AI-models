# AI-Models Repository

This repository houses a collection of AI models, including a simple linear regression model and several pre-trained models like ResNet50, InceptionV3, MobileNetV2, and VGG16.  It provides scripts for training, testing, and converting these models to TensorFlow Lite format for deployment on Android devices.

## Table of Contents

* [Overview](#overview)
* [Models](#models)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
    * [Training](#training)
    * [Conversion to TensorFlow Lite](#conversion-to-tensorflow-lite)
    * [Debug mode usage](#debugging)
* [File Structure](#file-structure)

## Overview

This project leverages TensorFlow to implement a variety of AI models, ranging from a basic linear regression to complex pre-trained architectures.  Models are saved in both Keras `.h5` format (for training and further development) and TensorFlow Lite `.tflite` format (optimized for mobile deployment).

## Models

The following models are included in this repository:

* **Linear Regression:** Trained on the data `y = x^2`.
* **ResNet50:** Pre-trained on the ImageNet dataset.
* **InceptionV3:** Pre-trained on the ImageNet dataset.
* **MobileNetV2:** Pre-trained on the ImageNet dataset.
* **VGG16:** Pre-trained on the ImageNet dataset.

## Prerequisites

Before getting started, ensure you have the following software installed:

* Python 3.10+
* TensorFlow
* NumPy
* Scikit-learn
* SciPy
* TensorFlow Hub

It is **highly recommended** to use a virtual environment to manage dependencies. Conda is a good option:

If you don’t have Conda installed, download and install Miniconda or Anaconda from:
https://docs.conda.io/en/latest/miniconda.html


## Installation

Clone the repository:
```bash
git clone https://github.com/Vishwatej1828/AI-models.git
```

Change directory to the repository
```bash
cd AI-models
```
Create and activate the Conda environment

```bash
conda env create -f ai_models_env.yml
conda activate ai-models-env
```

## Usage

The `scripts/model.py` script handles both training and conversion.

### Training
To train a model and convert it to TensorFlow Lite in a single step:
```Bash
    python scripts/model.py -m <model_name> --train -i <path_to_test_image>
```

Replace <model_name> with the name of the model (e.g., "ResNet50", "Linear-Regression"). Replace <path_to_test_image> with the path to an image used for testing (this might be used during the conversion process).


### Conversion to TensorFlow Lite

To convert an existing .h5 model to TensorFlow Lite format:
```bash
python scripts/model.py -m <model_name> -i <path_to_test_image> -c
```

### Debugging

For verbose output and debugging information, use the -v flag:
```
python scripts/model.py -m <model_name> -i <path_to_test_image> -c -v
```

## File Structure

Models are saved in the following directories:

    tf_models_dir/: Contains the Keras .h5 model files. For example:
        resnet50.h5
        inceptionV3.h5
        mobilenetV2.h5
        vgg16.h5
    tf_lite_models_dir/: Contains the TensorFlow Lite .tflite model files. For example:
        resnet50.tflite
        inceptionV3.tflite
        mobilenetV2.tflite
        vgg16.tflite
