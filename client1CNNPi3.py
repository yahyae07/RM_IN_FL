import os
import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load data
data = pd.read_csv("RPi3_datasets/dataPi3_C1.csv")
X = data.iloc[:, :-1].values  # Features (all columns except the last one)
y = data.iloc[:, -1].values   # Target variable (last column)

# Reshape the input data
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Flower client
class RegressionClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32)
        
        # Calculate and print MSE and R2 score on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("MSE:", mse)
        print("R2 Score:", r2)
        print("Accuracy:", r2 * 100, "%")
        
        # Save the model
        # model.save("model.h5")
        
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss = model.evaluate(X_test, y_test)
        return loss, len(X_test), {}

# Create CNN regression model using TensorFlow
model = keras.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer with single node for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()

# Get the number of weights in the model
num_weights = len(model.get_weights())
print("Number of weights in the model:", num_weights)

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=RegressionClient())
