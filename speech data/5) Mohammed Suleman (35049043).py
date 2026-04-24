#5) Mohammed Suleman (35049043)

# -----------------------------
# 17. Plot training curves
# -----------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.show()

# -----------------------------
# 18. Evaluate on test set
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average='weighted'
)

print(f"Test Accuracy : {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test F1-Score : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# -----------------------------
# 19. Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 20. Save label classes
# -----------------------------
np.save("speech_label_classes.npy", label_encoder.classes_)
print("Model and label classes saved.")

# -----------------------------
# 21. Single file prediction function
# -----------------------------
def predict_emotion(file_path, model, label_encoder):
    signal, sr = load_audio_fixed(file_path)
    mfcc = extract_mfcc(signal, sr)
    mfcc = np.expand_dims(mfcc, axis=0)  # batch dimension

    pred_probs = model.predict(mfcc, verbose=0)
    pred_idx = np.argmax(pred_probs, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return pred_label, pred_probs[0]

# Example:
sample_file = df.iloc[0]["path"]
pred_label, pred_probs = predict_emotion(sample_file, model, label_encoder)
print("Sample file:", sample_file)
print("Predicted emotion:", pred_label)