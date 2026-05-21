MAP_WIDTH = 50
MAP_HEIGHT = 50
GRID_RESOLUTION = 1

# =========================
# TASK SETTINGS
# =========================

NUM_TASKS = 100
HIGH_PRIORITY_RATIO = 0.3

# =========================
# UAV SETTINGS
# =========================

NUM_UAVS = 10

MIN_ENERGY = 300
MAX_ENERGY = 500

MIN_HOVER_TIME = 150
MAX_HOVER_TIME = 300

MIN_COMPUTE = 300
MAX_COMPUTE = 500

# =========================
# PARTITIONING PARAMETERS
# =========================

ALPHA = 1.0
GAMMA = 15.0
RHO = 0.05
LAMBDA_TV = 5.0
ITERATIONS = 20

# =========================
# RL PARAMETERS
# =========================

EPOCHS = 500
RL_ALPHA = 0.1
RL_GAMMA = 0.9
EPSILON = 0.2

# =========================
# DYNAMIC REPARTITIONING
# =========================

DELTA_T = 10
HYSTERESIS_THRESHOLD = 5.0
MAX_CELL_REASSIGNMENTS = 10