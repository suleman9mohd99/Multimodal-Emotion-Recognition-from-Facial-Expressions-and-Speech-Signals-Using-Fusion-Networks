

#1) Nikhilesh Andole (35040343)


from google.colab import drive
drive.mount('/content/drive')

# -----------------------------
# 1. Install required libraries
# -----------------------------
!pip install -q librosa soundfile tensorflow scikit-learn seaborn

# -----------------------------
# 2. Imports
# -----------------------------
# -----------------------
# System libraries
# -----------------------
import os
import zipfile
import shutil
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Data handling
# -----------------------
import numpy as np
import pandas as pd

# -----------------------
# Audio processing
# -----------------------
import librosa
import librosa.display
import soundfile as sf

# -----------------------
# Visualization
# -----------------------
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Machine learning tools
# -----------------------
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

# -----------------------
# TensorFlow / Deep Learning
# -----------------------
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Bidirectional,
    LSTM,
    Dense,
    GlobalAveragePooling1D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam

# -----------------------
# Print versions (optional)
# -----------------------
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# -----------------------------
# 3. GPU check
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus)
else:
    print("No GPU detected. In Colab: Runtime -> Change runtime type -> GPU")

# -----------------------------
# 4. Path define from drive
# -----------------------------

EXTRACT_DIR = "/content/drive/MyDrive/RAVDESS Emotional dataset"

# -----------------------------
# 5. RAVDESS label mapping
# Filename format:
# 03-01-06-01-02-01-12.wav
# 3rd part = emotion ID
# -----------------------------
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# -----------------------------
# 6. Build dataframe from files
# -----------------------------
data = []

for root, dirs, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            parts = file.replace(".wav", "").split("-")
            if len(parts) == 7:
                emotion_id = parts[2]
                actor_id = parts[6]

                emotion = emotion_map.get(emotion_id, "unknown")

                data.append({
                    "path": file_path,
                    "file_name": file,
                    "emotion_id": emotion_id,
                    "emotion": emotion,
                    "actor": actor_id
                })

df = pd.DataFrame(data)

print("Total files:", len(df))
print(df.head())

