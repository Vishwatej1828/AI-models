import numpy as np
import tensorflow as tf

def testLiteModel(testData):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="tf_lite_models/simple_linear_regression_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize an array to store predictions
    predictions = []
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

    print("=========================predictions by Tensorflow Lite Model=========================")
    # Print predictions alongside test data
    for test_value, prediction in zip(x_test, predictions):
        print(f"x_test: {test_value[0]:.2f} ---> Prediction: {prediction:.2f}")



def testOrigionalModel(testData):
    # Load the saved model
    model_directory = "saved_models_dir/simple_linear_regression_model_dir"
    model = tf.saved_model.load(model_directory)


    # Run predictions
    predictions = model(testData)

    # Print predictions
    print("=========================predictions by Origional Model=========================")
    for x, y_pred in zip(testData, predictions.numpy()):
        print(f"x: {x[0]:.2f}, predicted y: {y_pred[0]:.2f}")

# Prepare test data.   model trained on -10 to 10 but we are testing on the data 10 to 20
x_test = np.linspace(10, 20, 100)
x_test = x_test.reshape(-1, 1)  # Reshape for model's input format


testOrigionalModel(x_test)
testLiteModel(x_test)