import torch
from torch.utils.data import DataLoader

from dataloader import FRCDataset
from model import *

from eagle import Learner
from eagle.callback import *

import config

def main():
    class Pretrain:
        """
            Bootstrap (drive straight toward attacker) data
            Attacker is learned
            No extra objects
            Random field positions
        """
        train_loader = DataLoader(
            FRCDataset("/mnt/frc-rl/Data/pretrain/train"),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        valid_loader = DataLoader(
            FRCDataset("/mnt/frc-rl/Data/pretrain/valid"),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    # For now, we'll use the same data just on lower LR
    class Finetune:
        """
            Bootstrap (drive straight toward attacker) data
            Attacker is learned
            No extra objects
            Random field positions
        """
        train_loader = DataLoader(
            FRCDataset("/mnt/frc-rl/Data/finetune/train"),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        valid_loader = DataLoader(
            FRCDataset("/mnt/frc-rl/Data/finetune/valid"),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=1
        )

    model = Model()
    optim = torch.optim.AdamW(model.parameters())

    ############ PRETRAIN ############
    learner = Learner(model, optim, Pretrain.train_loader, Pretrain.valid_loader, loss_fn=FRCLoss(), clip_grads=1, autocast=False,
                    metrics = [LRMetric(), TrainLossMetric(), ValidLossMetric(), CorrectHitMetric(), VelocityPredictionMetric()],
                    cbs     = [SaveBestCallback("./exports/autosave"), StatusCallback()]
    )

    learner.fit_one_cycle(config.TRAIN_EPOCHS, max_lr=3e-4)
    learner.save("./exports/pretrain")

    ############ FINE TUNE ############
    learner = Learner(model, optim, Finetune.train_loader, Finetune.valid_loader, loss_fn=FRCLoss(), clip_grads=1, autocast=False,
                    metrics = [LRMetric(), TrainLossMetric(), ValidLossMetric(), CorrectHitMetric(), VelocityPredictionMetric()],
                    cbs     = [SaveBestCallback("./exports/autosave"), StatusCallback()]
    )

    learner.fit_one_cycle(config.TUNE_EPOCHS, max_lr=1e-4)
    learner.save("./exports/finetune")


if __name__ == "__main__":
    main()
