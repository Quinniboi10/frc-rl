import torch

from pathlib import Path
from dataclasses import dataclass
import logging

from dataloader import Datapoint
from model import Model
import config

# Use "./exports/finetune.pt" by default

@dataclass
class DefenseModelOutput():
    ray_hit_confidences: list[float] # Model output confidences post-sigmoid
    relative_velocity_forward: float # Relative to the local robot, not the detected robot
    relative_velocity_right: float # Relative to the local robot, not the detected robot

class DefenseInferenceStack():
    def __init__(self, path: Path):
        self.model = Model()
        self.model.load_state_dict(
            torch.load(path, weights_only=True, map_location="cpu")
        )
        self.model.eval()
        self.model = torch.compile(self.model)

    @torch.inference_mode()
    def __call__(self, frames: list[Datapoint]):
        try:
            # Get 10 frames ready
            frames = frames[-config.FRAMES:]
            if len(frames) < config.FRAMES:
                frames = [frames[0]] * (config.FRAMES - len(frames)) + frames

            # Build the input tensor
            in_tensor = torch.concat([frame.get_input_tensor() for frame in frames]).unsqueeze(0) # Add the batch dimension

            # Run inference
            out_tensor: torch.Tensor = self.model(in_tensor).squeeze(0) # Strip the batch dimension
            out_tensor[:-2] = out_tensor[:-2].sigmoid()

            if not torch.isfinite(out_tensor).all():
                raise RuntimeError("Got non-finite numbers!")

            # Build output format
            return DefenseModelOutput(out_tensor[:-2].tolist(), out_tensor[-2].item(), out_tensor[-1].item())
        except Exception: # Broadly catch errors here to avoid crashing from bad data or something
            logging.exception("Model inference failed")
            return DefenseModelOutput([0.0] * 65, 0.0, 0.0)
