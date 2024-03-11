RANDOM_SEED = 42

# Features after EDA
INPUT_SIZE = 14

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MIN_EPOCHS = 1
MAX_EPOCHS = 13
TRAIN_VAL_RATIO = 0.8

# Loggers
TENSORBOARD_LOGGER = False
MLFLOW_LOGGER = False

# Dataset
TRAIN_DATA_PATH = "data/custom_train_df.csv"
TEST_DATA_PATH = "data/custom_test_df.csv"
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 16

# Save Path
MODEL_PATH = "../app/model.pt"

# ML Flow
TRACKING_URL = "http://51.250.112.132:5000"
EXPERIMENT_NAME = "VK_TEST"
LOG_MODEL = False
