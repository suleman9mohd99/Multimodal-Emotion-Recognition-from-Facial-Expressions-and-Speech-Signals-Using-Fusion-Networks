#1. Ashiq Shaik (35033254)


# =========================
# 1. Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 2. Install required package
# =========================
!pip -q install xgboost

# =========================
# 3. Imports
# =========================
import os
import re
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

print("Libraries loaded successfully.")

# =========================
# 4. Set dataset folder path
# =========================
# Change this path if needed
dataset_dir = "/content/drive/MyDrive/Face dataset"

print("Dataset folder exists:", os.path.exists(dataset_dir))

# =========================
# 5. Find all CSV files recursively
# =========================
csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)

print("Total CSV files found:", len(csv_files))
print("Sample files:")
for f in csv_files[:5]:
    print(f)

# =========================
# 6. Emotion mapping from filename
# RAVDESS-style code:
# 01 = neutral
# 02 = calm
# 03 = happy
# 04 = sad
# 05 = angry
# 06 = fear
# 07 = disgust
# 08 = surprised
# =========================
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}

def get_emotion_from_filename(filepath):
    filename = os.path.basename(filepath).replace(".csv", "")
    parts = filename.split("-")
    if len(parts) >= 3:
        emotion_code = parts[2]
        return emotion_map.get(emotion_code, "unknown")
    return "unknown"

