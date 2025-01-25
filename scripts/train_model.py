import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import os
import math

# Generate data
x = np.linspace(-10, 10, 5000).reshape(-1, 1)  # Input data
y = (x ** 2).reshape(-1, 1)  # Target data

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
saved_model_dir = "saved_models_dir/simple_linear_regression_model_dir"
model.export(saved_model_dir)  # Export the model to the specified directory

print(f"Model saved in {os.path.abspath(saved_model_dir)}")