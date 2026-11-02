"""Central configuration for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

CLASS_NAMES = ["normal", "DoS", "Probe", "R2L", "U2R"]

SEED = 42
LATENT_DIM = 100
GAN_EPOCHS = 100
GAN_BATCH_SIZE = 128
CLF_EPOCHS = 30
CLF_BATCH_SIZE = 256
# Minority classes are oversampled up to TARGET_RATIO * (size of largest class)
TARGET_RATIO = 0.3
