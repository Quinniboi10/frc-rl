from pathlib import Path

EXPECTED_RAYS = 32 * 2 + 1

STACKING = 10

MIN_EPISODE_LENGTH = 20

TRAIN_DATA = Path("/mnt/frc-rl/Data/train")
VALID_DATA = Path("/mnt/frc-rl/Data/valid")

BATCH_SIZE = 128
EPOCHS = 15