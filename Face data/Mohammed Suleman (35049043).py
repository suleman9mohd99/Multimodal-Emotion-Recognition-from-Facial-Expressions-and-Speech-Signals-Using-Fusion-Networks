#2. Mohammed Suleman (35049043)

# =========================
# 7. Read one sample file
# =========================
sample_df = pd.read_csv(csv_files[0])
print("\nSample file shape:", sample_df.shape)
print("\nFirst 20 columns:")
print(sample_df.columns[:20].tolist())

# =========================
# 8. Select useful columns
# Fast + strong features:
# gaze, pose, AU, confidence, success
# =========================
all_cols = sample_df.columns.tolist()

selected_cols = []
for col in all_cols:
    if (
        col.startswith("gaze_") or
        col.startswith("pose_") or
        col.startswith("AU")
    ):
        selected_cols.append(col)

if "confidence" in all_cols:
    selected_cols.append("confidence")

if "success" in all_cols:
    selected_cols.append("success")

print("\nSelected feature columns:", len(selected_cols))
print(selected_cols[:20])

# =========================
# 9. Feature extraction
# For each CSV file:
# mean, std, min, max, median
# =========================
def extract_features_from_csv(filepath, selected_cols):
    try:
        df = pd.read_csv(filepath)

        cols = [c for c in selected_cols if c in df.columns]
        df = df[cols].copy()

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(0)

        feature_dict = {}

        for c in df.columns:
            values = df[c].values.astype(np.float32)

            feature_dict[f"{c}_mean"] = np.mean(values)
            feature_dict[f"{c}_std"] = np.std(values)
            feature_dict[f"{c}_min"] = np.min(values)
            feature_dict[f"{c}_max"] = np.max(values)
            feature_dict[f"{c}_median"] = np.median(values)

        return feature_dict

    except Exception as e:
        return None

# =========================
# 10. Build dataset
# =========================
data_rows = []

for i, file_path in enumerate(csv_files):
    label = get_emotion_from_filename(file_path)
    if label == "unknown":
        continue

    feats = extract_features_from_csv(file_path, selected_cols)
    if feats is None:
        continue

    feats["label"] = label
    feats["file_name"] = os.path.basename(file_path)
    feats["full_path"] = file_path
    data_rows.append(feats)

    if (i + 1) % 200 == 0:
        print(f"Processed {i+1}/{len(csv_files)} files")

dataset_df = pd.DataFrame(data_rows)

print("\nFinal dataset shape:", dataset_df.shape)
print("\nClass counts:")
print(dataset_df["label"].value_counts())

