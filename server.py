import flwr as fl


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=100),
)

# Run 5 CNN clients 10 rounds, MSE: 11.114, R2: 0.876, Accuracy: 87.6%
# poetry run python3 server.py & poetry run python3 client1CNN.py & poetry run python3 client2CNN.py & poetry run python3 client3CNN.py & poetry run python3 client4CNN.py & poetry run python3 client5CNN.py

# Run 10 CNN clients 10 rounds,
# poetry run python3 server.py & poetry run python3 client1CNN.py & poetry run python3 client2CNN.py & poetry run python3 client3CNN.py & poetry run python3 client4CNN.py & poetry run python3 client5CNN.py & poetry run python3 client6CNN.py & poetry run python3 client7CNN.py & poetry run python3 client8CNN.py & poetry run python3 client9CNN.py & poetry run python3 client10CNN.py

# Run 10 CNN clients, 10 rounds, 5 different devices
# poetry run python3 server.py & poetry run python3 client1CNN.py & poetry run python3 client2CNN.py & poetry run python3 client1CNNCoral.py & poetry run python3 client2CNNCoral.py & poetry run python3 client1CNNJetson.py & poetry run python3 client2CNNJetson.py & poetry run python3 client1CNNPi2.py & poetry run python3 client2CNNPi2.py & poetry run python3 client1CNNPi3.py & poetry run python3 client2CNNPi3.py

