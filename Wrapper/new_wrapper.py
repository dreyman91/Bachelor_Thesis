import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN


class ObservationWrapper:
    def __init__(self, mem_size=1000, clustering_eps=0.5, clustering_min_samples=3):
        self.mem_size = mem_size
        self.obs_mem = []
        input_obs = self.preprocess_observation({"id": 0, "position": [0, 0, 0], "velocity": [0, 0, 0], "rotation": [0, 0], "dynamic_state": 0})
        self.encoder = ObservationEncoder(input_dim=len(input_obs), latent_dim=len(input_obs))
        self.cluster_model = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples)

    def preprocess_observation(self, raw_obs): # Modify the preprocessing to handle variable legth
        object_id = raw_obs.get("id", -1)
        position = raw_obs.get("position", [0, 0, 0])
        velocity = raw_obs.get("velocity", [0, 0, 0])
        rotation = raw_obs.get("rotation", [0, 0])
        dynamic_state = raw_obs.get("dynamic_state", 0)

        return np.array([object_id] + position + velocity + rotation + [dynamic_state])

    def encode_features(self, processed_obs):
        tensor_obs = torch.tensor(processed_obs, dtype=torch.float32, requires_grad=False)
        encoded_features, _ = self.encoder(tensor_obs)
        return encoded_features.detach().numpy().flatten()

    def cluster_observation(self):
        if len(self.obs_mem) == 0:
            return np.array([-1])
        if len(self.obs_mem) < self.cluster_model.min_samples:
            return np.full(len(self.obs_mem), -1)

        obs_array = np.array(self.obs_mem)
        cluster_labels = self.cluster_model.fit_predict(obs_array)
        return cluster_labels

    def manage_memory(self, new_encoded_obs):
        if any(np.allclose(new_encoded_obs, obs, atol=1e-3) for obs in self.obs_mem):
            return
        self.obs_mem.append(new_encoded_obs)
        if len(self.obs_mem) > self.mem_size:
            self.obs_mem.pop(0)

    def standardize_output(self, new_obs):
        processed_obs = self.preprocess_observation(new_obs)
        encoded_obs = self.encode_features(processed_obs)
        self.manage_memory(encoded_obs)
        clusters = self.cluster_observation()
        return np.concatenate([encoded_obs, [clusters[-1]]])


class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ObservationEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed
