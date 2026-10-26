import torch
from torch.utils.data import Dataset

from pathlib import Path

from typing import get_type_hints, get_origin, get_args
from dataclasses import dataclass, fields

import config

@dataclass
class Datapoint:
    step: int

    velDotFwd: float
    velDotRight: float

    posX: float
    posZ: float

    velAngular: float
    rotSin: float
    rotCos: float

    raycastDistances: list[float]

    hitsTarget: list[bool]
    targetVelDotFwd: float
    targetVelDotRight: float

    def __init__(self):
        pass

    def get_input_tensor(self) -> torch.Tensor:
        return torch.tensor([self.velDotFwd, self.velDotRight, self.posX, self.posZ, self.velAngular, self.rotSin, self.rotCos] + self.raycastDistances, dtype=torch.float32)

    def get_target_tensor(self) -> torch.Tensor:
        return torch.tensor(self.hitsTarget + [self.targetVelDotFwd, self.targetVelDotRight], dtype=torch.float32)

@dataclass
class Episode:
    frames: list[Datapoint]

    def __post_init__(self):
        assert len(self.frames) > 0

    # Returns config.STACKING datapoints in a list
    def __getitem__(self, idx: int) -> list[Datapoint]:
        assert(0 <= idx < len(self.frames))
        start = max(0, idx - config.STACKING)
        padding = max(0, config.STACKING - idx)
        return [self.frames[0]] * padding + self.frames[start:idx]


class FRCDataset(Dataset):
    def __init__(self, root: Path):
        files: list[Path] = [file for file in root.rglob("*.txt") if file.is_file()]

        self.datapoints: list[Datapoint] = []

        # Extract datapoints from all files as a list of Datapoint objects
        for file in files:
            for line in file.read_text().splitlines():
                dp = Datapoint()
                data = iter(line.split('|'))

                types = get_type_hints(dp)
                for field in fields(dp):
                    field_type = types[field.name]
                    if get_origin(field_type) == list:
                        # These will need to consume more than 1 token and save them all as 1 list
                        list_item_type = get_args(field_type)[0]
                        if list_item_type == bool:
                            value = [next(data) == "1" for _ in range(config.EXPECTED_RAYS)]
                        else:
                            value = [list_item_type(next(data)) for _ in range(config.EXPECTED_RAYS)]
                        setattr(dp, field.name, value)
                    else:
                        # Only 1 token will be consumed here
                        value = next(data)
                        setattr(dp, field.name, field_type(value))

                self.datapoints.append(dp)

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
        terminator.step = 0
        self.datapoints.append(terminator)

        # Extract episodes from each game
        for dp in self.datapoints:
            if len(dp_buffer) > 0 and dp.step <= dp_buffer[-1].step:
                if len(dp_buffer) > config.MIN_EPISODE_LENGTH:
                    for _ in range(len(dp_buffer)):
                        self.idx_to_ep.append(len(self.episodes))
                    self.idx_to_step.extend(range(len(dp_buffer)))
                    self.episodes.append(Episode(dp_buffer))
                dp_buffer = []
            dp_buffer.append(dp)

        print(f"Parsed {sum(file.stat().st_size / (1000 ** 2) for file in files):.2f} MB across {len(files)} file{"s" if len(files) > 1 else ""}, containing {len(self):,} usable datapoints with {len(self) / len(self.datapoints) * 100:.2f}% retention and {len(self.episodes)} episodes")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_idx = self.idx_to_ep[idx]
        frames = self.episodes[episode_idx][self.idx_to_step[idx]]

        assert len(frames) == config.STACKING, f"{idx} has a length of {len(frames)}"

        return torch.concat([frame.get_input_tensor() for frame in frames]), frames[-1].get_target_tensor()

    def __len__(self):
        return len(self.idx_to_ep)
