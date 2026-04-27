#4.Nikhilesh Andole (35040343)

# =========================
# 15. Train models and compare
# =========================
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    results.append((name, val_acc))
    print(f"{name} Validation Accuracy: {val_acc:.4f}")

results_df = pd.DataFrame(results, columns=["Model", "Validation Accuracy"])
results_df = results_df.sort_values(by="Validation Accuracy", ascending=False)

print("\nModel Comparison:")
print(results_df)

plt.figure(figsize=(8, 4))
sns.barplot(data=results_df, x="Model", y="Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

# =========================
# 16. Select best model
# =========================
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

print(f"\nBest model selected: {best_model_name}")

# Retrain on full train+validation data
best_model.fit(X_train_full, y_train_full)

