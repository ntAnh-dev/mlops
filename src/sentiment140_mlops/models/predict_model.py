from fastapi import FastAPI
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import joblib, tensorflow as tf

app = FastAPI(title="Sentiment140 API")

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert")
    model = TFDistilBertForSequenceClassification.from_pretrained("models/distilbert")
    model_type = "DistilBERT"
except:
    tokenizer, model = joblib.load("models/ml_model.pkl")
    model_type = "LogisticRegression"

@app.post("/predict")
def predict(text: str):
    if model_type == "DistilBERT":
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        label = int(tf.argmax(probs, axis=1))
        conf = float(tf.reduce_max(probs))
    else:
        label = model.predict([text])[0]
        conf = 1.0
    sentiment = "positive" if label == 1 else "negative"
    return {"sentiment": sentiment, "confidence": round(conf, 3)}
