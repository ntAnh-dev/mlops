import os
import pandas as pd
import subprocess

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RAW_FILE = f"{RAW_DIR}/training.1600000.processed.noemoticon.csv"
PROCESSED_FILE = f"{PROCESSED_DIR}/sentiment140.csv"

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "kazanova/sentiment140",
        "-p", RAW_DIR
    ], check=True)
    subprocess.run(["unzip", "-o", f"{RAW_DIR}/sentiment140.zip", "-d", RAW_DIR], check=True)

def preprocess():
    df = pd.read_csv(
        RAW_FILE, encoding="latin-1", header=None,
        names=["target", "ids", "date", "flag", "user", "text"]
    )
    df = df[["text", "target"]]
    df["label"] = df["target"].apply(lambda x: 1 if x == 4 else 0)
    df = df[["text", "label"]].dropna()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Processed dataset saved at {PROCESSED_FILE}, size={len(df)}")

if __name__ == "__main__":
    download_data()
    preprocess()
