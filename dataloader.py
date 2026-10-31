from __future__ import annotations

import torch
from torch.utils.data import Dataset

from pathlib import Path
import time

from typing import get_type_hints, get_origin
from dataclasses import dataclass, field, fields

import config

import numpy as np
from numpy.typing import NDArray

# Some variables are named in terms of Z because when training in Unity, Y represents up/down

@dataclass
class Frame:
    vel_fwd: float = field(init=False)
    vel_right: float = field(init=False)

    pos_x: float = field(init=False)
    pos_z: float = field(init=False)

    vel_angular: float = field(init=False)
    rot_sin: float = field(init=False)
    rot_cos: float = field(init=False)

    raycast_distances: list[float] = field(init=False)

    @classmethod
    def from_dict(cls, data: dict) -> Frame:
        frame = cls()

        assert set(f.name for f in fields(cls)) == set(data.keys()), "Attempted to construct Frame with bad values"

        exp_types = get_type_hints(frame)
        for f in fields(cls):
            exp_type = exp_types[f.name]
            setattr(frame, f.name, exp_type(data[f.name]))

        return frame

    def get_input_tensor(self) -> torch.Tensor:
        return torch.tensor([self.vel_fwd, self.vel_right, self.pos_x, self.pos_z, self.vel_angular, self.rot_sin, self.rot_cos] + self.raycast_distances, dtype=torch.float32)

@dataclass
class Target:
    hits_target: list[bool] = field(init=False)
    target_vel_fwd: float = field(init=False)
    target_vel_right: float = field(init=False)

    def get_target_tensor(self) -> torch.Tensor:
        return torch.tensor(self.hits_target + [self.target_vel_fwd, self.target_vel_right], dtype=torch.float32)

@dataclass
class Datapoint:
    step: int = field(init=False)
    input_frame: Frame
    target: Target

    def __init__(self):
        self.input_frame = Frame()
        self.target = Target()

@dataclass
class Episode:
    frames: list[Datapoint]

    def __post_init__(self):
        assert len(self.frames) > 0

    # Returns config.STACKING datapoints in a list
    def __getitem__(self, idx: int) -> list[Datapoint]:
        assert(0 <= idx < len(self.frames))
        start = max(0, idx - config.FRAMES)
        padding = max(0, config.FRAMES - idx)
        return [self.frames[0]] * padding + self.frames[start:idx]

def process_data(data: list[NDArray[np.float32]], inputs: NDArray[np.float32], targets: NDArray[np.float32], step: NDArray[np.int32]) -> np.int64:
    idx = np.int64(0)

    size_inputs  = inputs.shape[1]
    size_targets = targets.shape[1]

    episodes = np.int64(0)

    raw_step = np.zeros_like(step)

    for file in data:
        for line_idx in range(file.shape[0]):
            line = file[line_idx]

            curr_step = raw_step[idx] = line[0] 
            if idx == 0 or curr_step <= raw_step[idx - 1]:
                step[idx] = 0
                episodes += 1
            else:
                step[idx] = step[idx - 1] + 1 # Ensure step is linear throughout the episode
            
            inputs[idx] = line[1:size_inputs+1]
            targets[idx] = line[size_inputs+1:size_inputs+size_targets+1]
            idx += 1

    assert idx == inputs.shape[0], "Something went wrong while loading data; not all expected datapoints were processed"

    return episodes

class FRCDataset(Dataset):
    def __init__(self, files: list[Path]):
        """
        NDArray [index][inputs] - float32 -> input tensor
        NDArray [index][target] - float32 -> target tensor
        NDArray [index]         - int     -> steps (recomputed to count up by 1 OR go down to 0)
        """
        start = time.monotonic()

        data = [np.loadtxt(file, delimiter='|', dtype=np.float32, ndmin=2) for file in files]

        self.num_datapoints = sum(file.shape[0] for file in data)

        frame_types = get_type_hints(Frame)
        target_types = get_type_hints(Target)

        num_input_values = sum(config.EXPECTED_RAYS if get_origin(frame_types[field.name]) is list else 1 for field in fields(Frame))
        num_target_values = sum(config.EXPECTED_RAYS if get_origin(target_types[field.name]) is list else 1 for field in fields(Target))

        self.inputs  = np.zeros((self.num_datapoints, num_input_values), dtype=np.float32)
        self.targets = np.zeros((self.num_datapoints, num_target_values), dtype=np.float32)
        self.step    = np.zeros(self.num_datapoints, dtype=np.int32)

        parsed_episodes = process_data(data, self.inputs, self.targets, self.step)

        total_mb = sum(file.stat().st_size / (1000 ** 2) for file in files)
        print(f"Parsed {total_mb:,.2f} MB across {len(files):,} file{"s" if len(files) > 1 else ""}, containing {self.num_datapoints:,} usable datapoints across {parsed_episodes:,} episodes ({total_mb / (time.monotonic() - start):,.2f}MB/s)")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.step[idx]
        padding = max(0, config.FRAMES - step - 1)

        if padding:
            first_frame = np.expand_dims(self.inputs[idx - step], axis=0) # Add a leading 1 to reshape to (1, N) so concat works
            input_pad = np.repeat(first_frame, padding, axis=0)

            unpadded = config.FRAMES - padding
            input_no_pad = self.inputs[idx+1-unpadded:idx+1]

            full_input = np.concatenate((input_pad, input_no_pad))
        else:
            full_input = self.inputs[idx-config.FRAMES+1:idx+1]

        target = self.targets[idx]

        return torch.from_numpy(full_input), torch.from_numpy(target)

    def __len__(self):
        return self.num_datapoints
