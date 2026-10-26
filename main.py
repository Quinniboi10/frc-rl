import torch
from torch.utils.data import DataLoader

from dataloader import FRCDataset
from model import *

from eagle import Learner
from eagle.callback import *

import config

def main():
    train_loader = DataLoader(
        FRCDataset(config.TRAIN_DATA),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    valid_loader = DataLoader(
        FRCDataset(config.VALID_DATA),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=1
    )

    model = Model()
    optim = torch.optim.AdamW(model.parameters())
    learner = Learner(model, optim, train_loader, valid_loader, loss_fn=FRCLoss(), clip_grads=1,
                    metrics = [LRMetric(), TrainLossMetric(), ValidLossMetric(), CorrectHitMetric(), VelocityPredictionMetric()],
                    cbs     = [SaveBestCallback("./exports/autosave"), StatusCallback()]
    )

    learner.fit_one_cycle(config.EPOCHS, max_lr=3e-4)

    learner.save("./exports/model")

if __name__ == "__main__":
    main()
