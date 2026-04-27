#3. Pamulapati Mahesh Babu (35045493)

# =========================
# 11. Plot class distribution
# =========================
plt.figure(figsize=(10, 4))
sns.countplot(x="label", data=dataset_df, order=dataset_df["label"].value_counts().index)
plt.title("Emotion Class Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# =========================
# 12. Prepare X and y
# =========================
X = dataset_df.drop(columns=["label", "file_name", "full_path"])
y = dataset_df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
print(list(label_encoder.classes_))

# =========================
# 13. Train/Val/Test split
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# =========================
# 14. Define fast strong models
# =========================
models = {
    "ExtraTrees": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "XGBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

