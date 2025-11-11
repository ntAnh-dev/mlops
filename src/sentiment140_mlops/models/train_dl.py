import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import mlflow
from datasets import Dataset

# Load data
df = pd.read_csv("data/processed/sentiment140.csv").sample(50000, random_state=42)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(df_train).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(df_test).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training
training_args = TrainingArguments(
    output_dir="./models/distilbert",
    eval_strategy="steps", 
    eval_steps=500,                
    save_strategy="no",            
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_dir="./reports/logs",
    logging_steps=100,             
    report_to="none"      
)

mlflow.set_experiment("Sentiment_DistilBERT_PyTorch")
with mlflow.start_run(run_name="DistilBERT_PyTorch_Sentiment140"):
    mlflow.transformers.autolog()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    model.save_pretrained("models/distilbert")
    tokenizer.save_pretrained("models/distilbert")
    print("✅ DistilBERT (PyTorch) trained & saved!")
