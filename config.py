# =========================================================
# DMMP-PR-TSA CONFIGURATION
# Based on: "Learning enhanced scheduling and resource
# allocation for heterogeneous UAV swarms in edge
# assisted remote sensing"
# =========================================================

# ---------------------------------------------------------
# MAP / GRID SETTINGS
# ---------------------------------------------------------
MAP_WIDTH  = 50         # Grid units
MAP_HEIGHT = 50
GRID_RESOLUTION = 1      # metres per cell (Δr)

SEED = 5 # change this for different random scenarios
SEED_UAV = 7 # change this for different UAV fleet generation

# ---------------------------------------------------------
# TASK SETTINGS
# ---------------------------------------------------------
NUM_TASKS           = 10
HIGH_PRIORITY_RATIO = 0.2   # fraction with priority=1

# Priority levels (paper Table 9)
# 1 = critical  (deadline 800 s)
# 2 = important (deadline 1000 s)
# 3 = routine   (deadline 1200 s)
PRIORITY_DEADLINES = {1: 800, 2: 1000, 3: 1200}

# Task type flags  (paper eq 13 / Table 3)
# -1  acquisition-only (no compute)
#  0  acquisition + light compute
#  1  integrated acquisition+processing (compute-intensive)
TASK_TYPE_RATIO = {-1: 0.4, 0: 0.3, 1: 0.3}

# ---------------------------------------------------------
# UAV FLEET SETTINGS
# ---------------------------------------------------------
NUM_UAVS = 4

# Energy budget  (J)
MIN_ENERGY = 150
MAX_ENERGY = 300

# Hovering time budget  (s)
MIN_HOVER_TIME = 80
MAX_HOVER_TIME = 200

# Compute budget  (GHz·s)
MIN_COMPUTE = 0.0
MAX_COMPUTE = 80.0

UAV_SPEED = 5.0

# Energy consumed per unit distance  (J/m)
ENERGY_PER_METER = 3.0

# UAV type capacities matching paper Table 1
# type -1: long-endurance acquisition-only (no compute)
# type  0: balanced
# type  1: processing-capable (shorter endurance)
UAV_TYPE_MAX_FLIGHT = {-1: 220, 0: 180, 1: 140}   # s
UAV_TYPE_MAX_COMPUTE = {-1: 0.0,  0: 80.0, 1: 60.0}   # GHz·s

# Maximum flight range derived from speed × max flight time
# Used in constraint (9): ||p_u - p_i|| <= D^max_u
# (computed at runtime from uav.max_hover_time * UAV_SPEED)

# ---------------------------------------------------------
# REGION PARTITIONING  (D-module, eq 3–8)
# ---------------------------------------------------------
ALPHA      = 1.0 / MAP_WIDTH  # distance weight
GAMMA      = 4.0  # priority attraction
RHO        = 0.05  # Lagrange sub-gradient step  ρ
ETA        = 5.0  # future energy penalty weight
LAMBDA_TV  = 0.2   # TV regularisation  λ_TV
ITERATIONS = 10    # power-diagram iterations

# Periodic re-partitioning hysteresis (section after eq 8)
DELTA_T                = 10    # re-evaluate every Δt seconds
HYSTERESIS_THRESHOLD   = 5.0   # ε: skip if ΔJ < ε
MAX_CELL_REASSIGNMENTS = 10    # budget B

# ---------------------------------------------------------
# SOM PRE/RE-ASSIGNMENT  (PR-module, eq 15–26)
# ---------------------------------------------------------
SOM_ROWS        = 5          # competitive layer is 5×U
SOM_ITERATIONS  = 200        # R iterations (Algorithm 1)
C_PHI           = 3.0        # c_ϕ  task-type matching coeff
C_RES           = 6.0        # c_RES resource matching coeff
C_S             = 20.0       # c_s  node influence (SOM grid)
C_TIME          = 0.1        # c_time for Δtime penalty (eq 21)
C_COMP          = 0.1        # c_comp for Δcomp penalty (eq 23)
SOM_LEARN_RATE  = 0.5        # initial SOM learning rate

# ---------------------------------------------------------
# RL TASK SEQUENCE ADJUSTMENT  (TSA-module, eq 27–30)
# ---------------------------------------------------------
EPOCHS    = 500      # Q-learning episodes
RL_ALPHA  = 0.1      # α  learning rate
RL_GAMMA  = 0.9      # γ  discount factor
EPSILON   = 0.1      # ε-greedy exploration

# Reward coefficients (eq 29)
CD = -6.0 / MAP_WIDTH  # c_d  distance penalty  (negative → minimise)
CP =  1.0   # c_p  priority reward
CT =  1.0   # c_t  endurance-preservation reward
CC =  1.0   # c_c  compute-sufficiency reward

# ---------------------------------------------------------
# DYNAMIC EVENTS
# ---------------------------------------------------------
ENABLE_DYNAMIC_EVENTS   = True
NEW_TASK_ARRIVAL_RATE   = 0.1   # probability per time-step
UAV_FAILURE_PROBABILITY = 0.02 