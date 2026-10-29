import torch
from torch.utils.data import Dataset

from pathlib import Path

from typing import get_type_hints, get_origin, get_args
from dataclasses import dataclass, field, fields

import config

# Some variables are named in terms of Z because when training in Unity, Y represents up/down

@dataclass
class Frame:
    step: int = field(init=False)

    vel_fwd: float = field(init=False)
    vel_right: float = field(init=False)

    pos_x: float = field(init=False)
    pos_z: float = field(init=False)

    vel_angular: float = field(init=False)
    rot_sin: float = field(init=False)
    rot_cos: float = field(init=False)

    raycast_distances: list[float] = field(init=False)

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


class FRCDataset(Dataset):
    def __init__(self, files: list[Path]):

        datapoints: list[Datapoint] = []

        # Extract datapoints from all files as a list of Datapoint objects
        for file in files:
            for line in file.read_text().splitlines():
                dp = Datapoint()
                data = iter(line.split('|'))

                for obj in (dp.input_frame, dp.target):
                    types = get_type_hints(obj)
                    for field in fields(obj):
                        field_type = types[field.name]
                        if get_origin(field_type) == list:
                            # These will need to consume more than 1 token and save them all as 1 list
                            list_item_type = get_args(field_type)[0]
                            if list_item_type == bool:
                                value = [next(data) == "1" for _ in range(config.EXPECTED_RAYS)]
                            else:
                                value = [list_item_type(next(data)) for _ in range(config.EXPECTED_RAYS)]
                            setattr(obj, field.name, value)
                        else:
                            # Only 1 token will be consumed here
                            value = next(data)
                            setattr(obj, field.name, field_type(value))

                datapoints.append(dp)

                extra_elems = 0
                try:
                    while True:
                        next(data)
                        extra_elems += 1
                except StopIteration:
                    if extra_elems != 0:
                        raise RuntimeError(f"{file} failed to parse; found {extra_elems} extra values")

        # All episodes
        self.episodes: list[Episode] = []

        # Index of datapoint -> index into episodes
        # Note that the index != self.datapoint index, because some episodes will be skipped
        self.idx_to_ep: list[int] = []
        self.idx_to_step: list[int] = []

        dp_buffer: list[Datapoint] = []

        # Add a quick datapoint at the end just to clear the buffer out
        terminator = Datapoint()
        terminator.input_frame.step = 0
        datapoints.append(terminator)

        # Extract episodes from each game
        for dp in datapoints:
            if len(dp_buffer) > 0 and dp.input_frame.step <= dp_buffer[-1].input_frame.step:
                if len(dp_buffer) > config.MIN_EPISODE_LENGTH:
                    for _ in range(len(dp_buffer)):
                        self.idx_to_ep.append(len(self.episodes))
                    self.idx_to_step.extend(range(len(dp_buffer)))
                    self.episodes.append(Episode(dp_buffer))
                dp_buffer = []
            dp_buffer.append(dp)

        print(f"Parsed {sum(file.stat().st_size / (1000 ** 2) for file in files):.2f} MB across {len(files)} file{"s" if len(files) > 1 else ""}, containing {len(self):,} usable datapoints with {len(self) / len(datapoints) * 100:.2f}% retention and {len(self.episodes):,} episodes")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_idx = self.idx_to_ep[idx]
        datapoints = self.episodes[episode_idx][self.idx_to_step[idx]]

        assert len(datapoints) == config.FRAMES, f"{idx} has a length of {len(datapoints)}"

        return torch.concat([dp.input_frame.get_input_tensor() for dp in datapoints]), datapoints[-1].target.get_target_tensor()

    def __len__(self):
        return len(self.idx_to_ep)
