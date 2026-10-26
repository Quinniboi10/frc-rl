import torch
from torch import nn

from eagle.metric import ExpAvgMetric, State

import config

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(72 * config.STACKING, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 67)
        )

    def forward(self, x: torch.Tensor):
        x = self.stack(x)
        return x

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
