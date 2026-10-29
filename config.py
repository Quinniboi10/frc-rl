from pathlib import Path

EXPECTED_RAYS = 32 * 2 + 1

FRAMES = 10
FRAME_SIZE = 72

FOV_PER_SIDE = 47.5 # Degrees the raycasts go in both left and right. Equal to FOV / 2

MIN_EPISODE_LENGTH = 20

BATCH_SIZE = 128
EPOCHS = 15

# CONTROLLER
TIMESTEP = 100 # ms

SEARCH_LOOKAHEAD = 0.3 # seconds, must be multiple of timestep
NUM_SEARCH_INTERVALS = 10

EST_BUPER_TO_CENTER = 0.5 # meters

class BLOCKING:
    MAX_Y_ERROR = 0.3 # m, horizontal error
    MAX_X_DISTANCE = 1 # m, sensor-to-opp-center

MAX_DROPOUT_TICKS = 2 # maximum number of ticks to continue blocking/intercepting on old paths before returning to search

CONF_THRESHOLD = 0.5