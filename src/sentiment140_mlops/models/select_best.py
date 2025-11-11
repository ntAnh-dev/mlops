import mlflow

client = mlflow.tracking.MlflowClient()
best = None
best_acc = 0.0

for exp in mlflow.search_experiments():
    runs = mlflow.search_runs([exp.experiment_id])
    if "metrics.accuracy" in runs.columns:
        top = runs.sort_values("metrics.accuracy", ascending=False).iloc[0]
        if top["metrics.accuracy"] > best_acc:
            best_acc = top["metrics.accuracy"]
            best = top

print(f"🏆 Best model: {best['tags.mlflow.runName']} with acc={best_acc:.4f}")
