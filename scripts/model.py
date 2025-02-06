import argparse
import os
import sys
import json
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3, VGG16

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_input
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array


# define model names to save the .h5 and .tflite
lr_name            = "linear_regression"
resnet_name        = "resnet50"
inception_name     = "inceptionV3"
mobilenet_name     = "mobilenetV2"
vgg16_name         = "vgg16"
tf_models_dir      = "tf_models_dir"
tf_lite_models_dir = "tf_lite_models_dir"

def setup_logging(verbose: bool):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=log_level
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and test models (Linear Regression or ResNet50 or MobileNetV2 or InceptionV3 or VGG16).",
        epilog="Example: python script.py --model ResNet50 --input-image /path/to/image.jpg -c --train -v"
    )
    parser.add_argument(
        "-m", "--model-name",
        type=str,
        required=True,
        choices=["Linear-Regression", "ResNet50", "MobileNetV2", "InceptionV3", "VGG16"],
        help="Name of the model to be trained (Linear-Regression or ResNet50 or InceptionV3)."
    )
    parser.add_argument(
        "-i", "--input-image",
        type=str,
        default="test_images/dog.jpg",
        help="Path to the input image for testing ResNet50."
    )

    parser.add_argument(
        "-c", "--convert",
        action="store_true",
        help="convert the model to tensorflow lite model."
    )

    parser.add_argument(
        "-tr", "--train",
        action="store_true",
        default=False,
        help="way of handling the model whether want to train."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    args = parser.parse_args()

    if (args.model_name == "ResNet50" or args.model_name == "InceptionV3" or args.model_name == "VGG16" or args.model_name == "MobileNetV2") and not args.input_image:
        parser.error(f"{args.model_name} requires an input image. Use -i to specify the path.")

    return args


def generate_linear_regression_data():
    """Generate training and testing data for linear regression."""
    x = np.linspace(-10, 10, 5000).reshape(-1, 1)
    y = (x ** 2).reshape(-1, 1)
    return train_test_split(x, y, test_size=0.2, random_state=42)


def build_linear_regression_model():
    """Build and compile a linear regression model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(1, activation='linear')
    ])
    logging.debug("compiling the Linear Regression Model")
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model(model, x_train, y_train):
    """Train the model."""
    logging.debug("Training the Linear Regression Model")
    model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0)
    logging.info("Model training complete.")


def save_model(model, model_name):
    """Save the model."""
    os.makedirs(tf_models_dir, exist_ok=True)
    model_path = os.path.join(tf_models_dir, f"{model_name}.h5")

    logging.debug(f"{model_name} is about to save")
    # Save the model in .h5 format
    model.save(model_path)
    logging.info(f"Saved model is: {os.path.abspath(model_path)}")


def load_model(model_name):
    """Load a saved model."""
    model_path = os.path.join(tf_models_dir, f"{model_name}.h5")

    logging.info(f"loading the model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model: {model_path} not found.")
    if model_name == "linear_regression":
        # as TensorFlow internally saves "mse" as a reference rather than the full function
        # When loading, it can't resolve "mse" unless explicitly passed
        return tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path, model_name, model):

    target_size = model.input_shape[1:3]
    logging.debug(f"input shape required for {model_name} is: {target_size}")

    """Load and preprocess an image for all models."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    if model_name == "ResNet50":
        img_array = preprocess_resnet_input(img_array)
    elif model_name == "InceptionV3":
        img_array = preprocess_inception_input(img_array)
    elif model_name == "MobileNetV2":
        img_array = preprocess_mobilenetv2_input(img_array)
    elif model_name == "VGG16":
        img_array = preprocess_vgg16_input(img_array)

    logging.debug(f"preprocessing of input data for model: {model_name} is done")
    return np.expand_dims(img_array, axis=0)


def test_model(model, model_name, test_data):
    logging.debug(f"testing the model: {model_name}")
    """Test the model with provided data."""
    if model_name == "Linear-Regression":
        predictions = model.predict(test_data)
        logging.info("Linear Regression Predictions:")
        for x, y_pred in zip(test_data, predictions):
            logging.info(f"x: {x[0]:.2f}, predicted y: {y_pred[0]:.2f}")
    elif model_name == "ResNet50" or model_name == "InceptionV3" or model_name == "MobileNetV2" or model_name == "VGG16":
        predictions = model.predict(test_data)
        logging.debug(f"model: {model_name} prediction is done")
        with open("resources/imagenet_class_index.json", "r") as f:
            class_index = json.load(f)
            logging.debug("class names are loaded")
        decoded_predictions = [
            class_index[str(pred.argmax())][1] for pred in predictions
        ]

        logging.info(f"=========================predictions by Tensorflow {model_name} Model=========================")
        for label in decoded_predictions:
            logging.info(f"Predicted Class: {label}")

def convert_to_lite_model(model, model_name):
    logging.debug(f"{model_name}: is about to convert to tflite model")
    # Convert the model to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    os.makedirs(tf_lite_models_dir, exist_ok=True)
    lite_model_name = os.path.join(tf_lite_models_dir, f"{model_name}.tflite")

    # Save the .tflite model.
    with open(lite_model_name, 'wb') as f:
        f.write(tflite_model)

    logging.info(f"{model_name} converted to TensorFlow Lite format.")


def test_lite_model(model_name, testData):
    logging.debug(f"testing the lite model: {model_name}: ")
    lite_model_path = ""
    if model_name == "Linear-Regression":
        lite_model_path = os.path.join(tf_lite_models_dir, f"{lr_name}.tflite")
    elif(model_name == "ResNet50"):
        lite_model_path = os.path.join(tf_lite_models_dir, f"{resnet_name}.tflite")
    elif (model_name == "InceptionV3"):
        lite_model_path = os.path.join(tf_lite_models_dir, f"{inception_name}.tflite")
    elif (model_name == "MobileNetV2"):
        lite_model_path = os.path.join(tf_lite_models_dir, f"{mobilenet_name}.tflite")
    elif model_name == "VGG16":
        lite_model_path = os.path.join(tf_lite_models_dir, f"{vgg16_name}.tflite")
    else:
        raise RuntimeError(f"{lite_model_path}: lite model is not exist")

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=lite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize an array to store predictions
    predictions = []

    if model_name == "Linear-Regression":
        # Loop through test data and get predictions
        for test_sample in testData:
            # Prepare the input data
            input_data = np.array([test_sample], dtype=np.float32)

            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get the output
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Store the prediction
            predictions.append(output_data[0][0])

        logging.info(f"=========================predictions by Tensorflow {model_name}Lite Model=========================")
        # Print predictions alongside test data
        for test_value, prediction in zip(testData, predictions):
            logging.info(f"x_test: {test_value[0]:.2f} ---> Prediction: {prediction:.2f}")

    elif model_name == "ResNet50" or model_name == "InceptionV3" or model_name == "MobileNetV2" or model_name == "VGG16":
        interpreter.set_tensor(input_details[0]['index'], testData)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Load ImageNet Class Labels**
        with open("resources/imagenet_class_index.json", "r") as f:
            class_index = json.load(f)
        decoded_predictions = [
            class_index[str(pred.argmax())][1] for pred in predictions
        ]

        logging.info(f"=========================predictions by Tensorflow Lite: {model_name} Model=========================")
        for label in decoded_predictions:
            logging.info(f"Predicted Class: {label}")


def main():
    """Main function to execute the script."""
    try:
        args = parse_arguments()
        setup_logging(args.verbose)

        # Initialize the model variable early
        model = None

        if args.model_name == "Linear-Regression":
            x_train, x_test, y_train, y_test = generate_linear_regression_data()

            # If the model exists and train is not provided then load the model
            if args.train or (not os.path.exists(os.path.join(tf_models_dir, f"{lr_name}.h5"))):
                model = build_linear_regression_model()
                train_model(model, x_train, y_train)
                save_model(model, lr_name)

            else:
                logging.info(f"Loading saved model {args.model_name}...")
                model =  load_model(lr_name)

            test_model(model, "Linear-Regression", x_test)
            if args.convert:
                convert_to_lite_model(model, lr_name)
                test_lite_model(args.model_name, y_test)


        elif args.model_name == "ResNet50":
            # If the model exists and train is not provided then load the model
            if args.train or (not os.path.exists(os.path.join(tf_models_dir, f"{resnet_name}.h5"))):
                model = ResNet50(weights="imagenet")
                save_model(model, resnet_name)

            else:
                logging.info(f"Loading saved model {args.model_name}...")
                model =  load_model(resnet_name)

            test_data = preprocess_image(args.input_image, args.model_name, model)
            logging.info(f"test_Data shape: {test_data.shape}")

            test_model(model, "ResNet50", test_data)
            if args.convert:
                convert_to_lite_model(model, resnet_name)
                test_lite_model(args.model_name, test_data)


        elif args.model_name == "InceptionV3":
            # If the model exists and train is not provided then load the model
            if args.train or (not os.path.exists(os.path.join(tf_models_dir, f"{inception_name}.h5"))):
                model = InceptionV3(weights="imagenet")
                save_model(model, inception_name)
            else:
                logging.info(f"Loading saved model {args.model_name}...")
                model =  load_model(inception_name)


            test_data = preprocess_image(args.input_image, args.model_name, model)
            logging.info(f"test_Data shape: {test_data.shape}")

            test_model(model, "InceptionV3", test_data)
            if args.convert:
                convert_to_lite_model(model, inception_name)
                test_lite_model(args.model_name, test_data)


        elif args.model_name == "MobileNetV2":
            # If the model exists and train is not provided then load the model
            if args.train or (not os.path.exists(os.path.join(tf_models_dir, f"{mobilenet_name}.h5"))):
                model = MobileNetV2(weights="imagenet")
                save_model(model, mobilenet_name)
            else:
                logging.info(f"Loading saved model {args.model_name}...")
                model =  load_model(mobilenet_name)

            test_data = preprocess_image(args.input_image, args.model_name, model)
            logging.info(f"test_Data shape: {test_data.shape}")

            test_model(model, "MobileNetV2", test_data)
            if args.convert:
                convert_to_lite_model(model, mobilenet_name)
                test_lite_model(args.model_name, test_data)

        elif args.model_name == "VGG16":
            # If the model exists and train is not provided then load the model
            if args.train or (not os.path.exists(os.path.join(tf_models_dir, f"{vgg16_name}.h5"))):
                model = VGG16(weights="imagenet")
                save_model(model, vgg16_name)
            else:
                logging.info(f"Loading saved model {args.model_name}...")
                model =  load_model(vgg16_name)


            test_data = preprocess_image(args.input_image, args.model_name, model)
            print(f"test_Data shape: {test_data.shape}")

            test_model(model, "VGG16", test_data)
            if args.convert:
                convert_to_lite_model(model, vgg16_name)
                test_lite_model(args.model_name, test_data)


    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
