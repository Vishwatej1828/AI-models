import tensorflow as tf

saved_model_dir = "saved_models_dir/simple_linear_regression_model_dir"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('tf_lite_models/simple_linear_regression_model.tflite', 'wb') as f:
  f.write(tflite_model)

print("Model converted to TensorFlow Lite format.")