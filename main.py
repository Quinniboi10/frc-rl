import torch
from torch.utils.data import DataLoader

from pathlib import Path
import sqlite3

from dataloader import FRCDataset
from model import *

from eagle import Learner
from eagle.callback import *

import config

"""
Training data should be organized in the .../<phase>/<train/valid>/.../*.txt format
The 'train' or 'valid' folder name must appear exactly once per tree
For instance, /mnt/frc/data/pretrain/train/file01.txt
"""

def main():
    DATA_PATHS = [
        Path("/mnt/frc-rl/Data/pretrain-bootstrap-mixed-field/train"),
        Path("/mnt/frc-rl/Data/pretrain-bootstrap-mixed-field/valid"),
        Path("/mnt/frc-rl/Data/finetune/train"),
        Path("/mnt/frc-rl/Data/finetune/valid"),
    ]

    SQL_PRETRAIN_TRAIN = "SELECT file FROM files WHERE phase GLOB 'pretrain*' AND split = 'train'"
    SQL_PRETRAIN_VALID = "SELECT file FROM files WHERE phase GLOB 'pretrain*' AND split = 'valid'"

    # Intentionally merge all data into 1 big batch
    SQL_FINETUNE_TRAIN = "SELECT file FROM files WHERE split = 'train'"
    SQL_FINETUNE_VALID = "SELECT file FROM files WHERE split = 'valid'"

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE files (file TEXT, phase TEXT, split TEXT);")
    for root in DATA_PATHS:
        root = root.resolve(strict=True)
        for file in root.rglob("*.txt"):
            parts = file.parts
            state = "train" if "train" in parts else "valid"
            phase = parts[parts.index(state) - 1]
            db.execute("INSERT INTO files (file, phase, split) VALUES (?, ?, ?)", (file.resolve().as_posix(), phase, state))

    print(f"Loaded {db.execute("SELECT COUNT(1) FROM files").fetchone()[0]} files across {db.execute("SELECT COUNT(DISTINCT phase) FROM files").fetchone()[0]} phases into the database")

    def get_files(cmd: str) -> list[Path]:
        return [Path(file[0]) for file in db.execute(cmd).fetchall()]

    class Pretrain:
        """
            Bootstrap (drive straight toward attacker) data
            Attacker is learned
            No extra objects
            Random field positions
        """
        train_loader = DataLoader(
            FRCDataset(get_files(SQL_PRETRAIN_TRAIN)),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        valid_loader = DataLoader(
            FRCDataset(get_files(SQL_PRETRAIN_VALID)),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    class Finetune:
        """
            Bootstrap (drive straight toward attacker) data
            Attacker is learned
            2026 field
            Semi-random field positions
        """
        train_loader = DataLoader(
            FRCDataset(get_files(SQL_FINETUNE_TRAIN)),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        valid_loader = DataLoader(
            FRCDataset(get_files(SQL_FINETUNE_VALID)),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    model = Model()
    optim = torch.optim.AdamW(model.parameters())

    ############ PRETRAIN ############
    learner = Learner(model, optim, Pretrain.train_loader, Pretrain.valid_loader, loss_fn=nn.BCEWithLogitsLoss(), clip_grads=1, autocast=False,
                    metrics = [LRMetric(), TrainLossMetric(), ValidLossMetric(), CorrectHitMetric()],
                    cbs     = [SaveBestCallback("./exports/autosave"), StatusCallback()]
    )

    learner.fit_one_cycle(config.TRAIN_EPOCHS, max_lr=3e-4)
    learner.save("./exports/pretrain")

    ############ FINE TUNE ############
    learner = Learner(model, optim, Finetune.train_loader, Finetune.valid_loader, loss_fn=nn.BCEWithLogitsLoss(), clip_grads=1, autocast=False,
                    metrics = [LRMetric(), TrainLossMetric(), ValidLossMetric(), CorrectHitMetric()],
                    cbs     = [SaveBestCallback("./exports/autosave"), StatusCallback()]
    )

    learner.fit_one_cycle(config.TUNE_EPOCHS, max_lr=1e-4)
    learner.save("./exports/finetune")


if __name__ == "__main__":
    main()
