# AI-models Repository
This repository contains AI models, including a simple linear regression model, and the necessary scripts for training, testing, and converting the model to TensorFlow Lite format for use in Android apps.


## Overview

- **Model**: A simple linear regression model is implemented using TensorFlow and trained on generated data.
- **Format**: The model is saved in both Keras `.h5` format (for training) and TensorFlow Lite `.tflite` format (for use in Android apps).


## Prerequisites

To use this repository, you need to have the following tools installed:

- **Python 3.8+**
- **TensorFlow** (for training and converting the model)
- **NumPy** (for data handling)
- **Scikit-learn** (for data splitting)
- **Matplotlib** (optional, for plotting data)
- **TensorFlow Lite** (for model conversion and testing)


## how to use the repo
1. clone the repository
    git clone https://github.com/Vishwatej1828/AI-models.git

2. Setting Up the Environment, using conda environment (Recommended)
    cd AI-models
    # create the environment
    conda env create -f ai_models_env.yml

    # activate the environment
    conda activate ai-models-env


## How to Train the Model
1. Clone the repository.
2. Set up the environment
3. Run `train_model.py` to train the model and convert it to TensorFlow Lite format.

    ```bash
    python scripts/train_model.py


# Converting the Model to TensorFlow Lite
    ```bash
    python scripts/convert_model.py

# how to test the model
Run `test_model.py` to test the model and TensorFlow Lite model

    ```bash
    python scripts/test_model.py


## saved model structure
saved_model_dir/
    assets/               # (optional, may be empty)     -> Contains additional files used by the model, such as vocabularies or feature dictionaries.
    saved_model.pb        # Contains the model's architecture and configuration, The protocol buffer file that stores the model's architecture, metadata, and training configuration.
    variables/
        variables.data-*  # Contains the model trained weights
        variables.index   # Index file for the weights vheckpoints