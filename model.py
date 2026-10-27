import torch
from torch import nn

from eagle.metric import ExpAvgMetric, State

import config

class Model(nn.Module):
    """
    Input rays are expected to be ordered from right (positive) to left (negative)
    """
    def __init__(self, channels=64, own_hidden=128, velocity_hidden=128):
        super().__init__()
        self.register_buffer('ray_coordinate', torch.linspace(1, -1, 65).reshape(1, 1, 65)) # Normalized angle
        self.own_embedding = nn.Sequential(
            nn.Linear(config.FRAMES * 7, own_hidden),
            nn.ReLU(),
            nn.Linear(own_hidden, channels)
        )
        self.range_embedding = nn.Conv1d(config.FRAMES + 1, channels, 5, padding=2)
        self.spatial = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.ReLU()
        )
        self.ray_output = nn.Conv1d(channels,1,1)
        self.velocity_output = nn.Sequential(
            nn.Linear(channels * 65, velocity_hidden),
            nn.ReLU(),
            nn.Linear(velocity_hidden, 2)
        )

    def forward(self, x: torch.Tensor):
        frames = x.reshape(-1, config.FRAMES, config.FRAME_SIZE)
        own = frames[:,:,:7].flatten(1) # Data representing info about the robot (flattened across time)
        ranges = frames[:,:,7:]         # Data representing info about the depth (not flattened)
        angles = ranges[:,:1,:]*0 + self.ray_coordinate # pyright: ignore[reportOperatorIssue] 

        # Convert the inputs into embeddings/internal state
        features: torch.Tensor = self.range_embedding(torch.cat([ranges,angles], dim=1))
        features = self.spatial(features+self.own_embedding(own).unsqueeze(-1))

        # Extract outputs from internal representation
        ray_logits = self.ray_output(features).flatten(1)
        velocity = self.velocity_output(features.flatten(1))

        return torch.cat([ray_logits,velocity], dim=1)

class FRCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.bce(x[..., :-2], y[..., :-2]) + self.mse(x[..., -2:], y[..., -2:])

class CorrectHitMetric(ExpAvgMetric):
    state = State.VALID
    do_reporting = True
    name = "hits accuracy"

    def step(self):
        self.avg.step(((self.learner.preds[..., :-2] > 0) == (self.learner.y[..., :-2]).bool()).float().mean().detach().item() * 100)


class VelocityPredictionMetric(ExpAvgMetric):
    state = State.VALID
    do_reporting = True
    name = "velocity prediction"

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def step(self):
        self.avg.step(self.mse(self.learner.preds[..., -2:], self.learner.y[..., -2:]).sqrt().detach())
