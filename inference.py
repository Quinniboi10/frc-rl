import torch

from pathlib import Path
from dataclasses import dataclass
from collections import deque
import logging
import time

from dataloader import Frame
from model import Model
import config

# Use "./exports/finetune.pt" by default

@dataclass
class DefenseModelOutput():
    ray_hit_confidences: list[float] # Model output confidences post-sigmoid
    relative_velocity_forward: float # Relative to the local robot, not the detected robot
    relative_velocity_right: float # Relative to the local robot, not the detected robot

class DefenseInferenceModel():
    def __init__(self, path: Path):
        self.model = Model()
        self.model.load_state_dict(
            torch.load(path, weights_only=True, map_location="cpu")
        )
        self.model.eval()
        self.model = torch.compile(self.model)

    @torch.inference_mode()
    def __call__(self, frames: list[Frame]) -> DefenseModelOutput:
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

class DefenseInferenceStack():
    MAX_FRAME_DRIFT = 15 # ms

    def __init__(self, model_path: Path):
        self.model = DefenseInferenceModel(model_path)
        self.frames: deque[Frame] = deque(maxlen=config.FRAMES)
        self.last_frame: float | None = None

    def add_frame(self, frame: Frame):
        now = time.monotonic()
        if self.last_frame is not None:
            time_since_last = (now - self.last_frame) * 1000 # ms
            if abs(time_since_last - config.TIMESTEP) > self.MAX_FRAME_DRIFT:
                logging.warning(f"Got frame {round(time_since_last)}ms after last which falls outside the window of 100ms +- {self.MAX_FRAME_DRIFT}")
        self.last_frame = now
        self.frames.append(frame)

    def predict(self):
        return self.model(list(self.frames))

    def reset(self):
        self.frames.clear()
