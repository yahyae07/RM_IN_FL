import os
import flwr as fl
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load data
data = pd.read_csv("RPi4_datasets/dataPi4_C5.csv")
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target variable (last column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Flower client
class RegressionClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss = model.evaluate(X_test, y_test)
        return loss, len(X_test), {}


# Create regression model using TensorFlow
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
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