import pandas as pd, tensorflow as tf, mlflow
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

df = pd.read_csv("data/processed/sentiment140.csv").sample(50000, random_state=42)
texts, labels = df["text"].tolist(), df["label"].tolist()

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
X = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
dataset = tf.data.Dataset.from_tensor_slices((dict(X), labels))
train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size).batch(16)
test_ds = dataset.skip(train_size).batch(16)

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

mlflow.set_experiment("Sentiment_DistilBERT")
with mlflow.start_run(run_name="DistilBERT_Sentiment140"):
    mlflow.transformers.autolog()
    model.compile(optimizer="adam", loss=model.compute_loss, metrics=["accuracy"])
    model.fit(train_ds, validation_data=test_ds, epochs=2)
    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    print("✅ DistilBERT trained & saved!")
