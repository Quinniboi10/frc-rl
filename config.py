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

MIN_SEARCH_LOOKAHEAD = 0.1 # sec
MAX_SEARCH_LOOKAHEAD = 5.0 # sec
INTERCEPT_OFFSET = 1.0 # meters between the opponent and the goal to try to reach
NUM_SEARCH_INTERVALS = 10
INTERCEPT_MAX_OMEGA = 0.1 # rad/s

class BLOCKING:
    MAX_Y_ERROR = 0.3 # m
    MAX_X_DISTANCE = 0.3 # m

CONF_THRESHOLD = 0.5