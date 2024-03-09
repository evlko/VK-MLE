# Features after EDA
INPUT_SIZE = 14

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MIN_EPOCHS = 1
MAX_EPOCHS = 10
TRAIN_VAL_RATIO = 0.8

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
