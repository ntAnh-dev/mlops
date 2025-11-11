import pandas as pd, joblib, mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/processed/sentiment140.csv").sample(50000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300)
with mlflow.start_run(run_name="ML_LogisticRegression"):
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    joblib.dump((vectorizer, model), "models/ml_model.pkl")
    print(f"✅ LogisticRegression acc={acc:.4f}")
