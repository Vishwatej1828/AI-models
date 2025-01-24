import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import math

"""
Tf Model saving
saved_model_dir/
├── assets/               # (optional, may be empty)     -> Contains additional files used by the model, such as vocabularies or feature dictionaries.
├── saved_model.pb        # Contains the model's architecture and configuration
├── variables/
│   ├── variables.data-*  # Contains the trained weights
│   └── variables.index   # Index file for the weights

Contents of the saved_model_dir

    saved_model.pb or saved_model.pbtxt: The protocol buffer file that stores the model's architecture, metadata, and training configuration.
    variables/:                          A folder containing:
        variables.data-*:                The model's trained weights.
        variables.index:                 An index file for the weight checkpoint.

"""

# Generate data
x = np.linspace(-10, 10, 5000).reshape(-1, 1)  # Input data
y = (x ** 2).reshape(-1, 1)  # Target data

"""# Plot the data
plt.plot(x, y)
plt.title("Quadratic Data (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""

# Split into training and testing data (random split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Build a simple regression model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),  # Hidden layer
    layers.Dense(1, activation='linear')                   # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0)

# Save the model in the SavedModel format
saved_model_dir = "saved_model_dir"
model.export(saved_model_dir)  # Export the model to the specified directory

print(f"Model saved in {os.path.abspath(saved_model_dir)}")


# convert a SavedModel into a TensorFlow Lite model.

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('simple_linear_regression_model.tflite', 'wb') as f:
  f.write(tflite_model)


# Evaluate the model on test data
loss, mae = model.evaluate(x_test, y_test)

# Print the results
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE (Mean Absolute Error): {mae}")


# Predict on the test data
y_pred = model.predict(x_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
