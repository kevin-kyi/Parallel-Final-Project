import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.autoencoder import Autoencoder
from utils.data_loader import load_features

import numpy as np

def load_features(features_path):
    try:
        features = np.load(features_path)  # Load the .npy file
        return features
    except Exception as e:
        raise IOError(f"Error loading features from {features_path}: {e}")


def train_autoencoder(features_path, model_path, input_dim, latent_dim, epochs=50, batch_size=64):
    features = load_features(features_path)
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch in loader:
            batch = batch[0]
            latent, reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    path = "data/processed/features.npy" #change to actual path of feature vecs
    train_autoencoder(path, "models/trained_model.pth", input_dim=300, latent_dim=32)
