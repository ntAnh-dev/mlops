# sentiment140_mlops

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

End-to-end MLOps pipeline built using **Cookiecutter Data Science v2 architecture**.

## Objectives
Predict tweet sentiment (positive / negative) using:
- Logistic Regression (TF-IDF)
- DistilBERT (Transformers)

## Tools
CookiecutterDataScience v2 · DVC · Git · MLflow · FastAPI · Docker · GitHub Actions · HuggingFace · Kaggle API

## Workflow
1. Download Sentiment140 via Kaggle API  
2. Preprocess & version data with DVC  
3. Train ML and DL models  
4. Track results with MLflow  
5. Serve with FastAPI  
6. Automate via GitHub Actions

## Running
### Pre-requisites
- [Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
- [tf-keras](https://pypi.org/project/tf-keras/) for DL
- Kaggle API: download and add Kaggle API Token
```
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

```
### Create environment
```
make create_environment
```
### Install package
```
make requirements
```
### Download and preprocess data
```
make download
```
### DVC
```
dvc add data/raw data/processed
git add data/.gitignore data/*.dvc
git commit -m "Add Sentiment140 data"
dvc push

dvc pull
```
### Train ML (TF-IDF + Logistic Regression)
```
make train_ml
```
### Train DL (DistilBERT)
```
make train_dl
```
### Select better model
```
make select
```
### Run MLflow (http://127.0.0.1:5000)
```
mlflow ui
```
### Run REST API (http://127.0.0.1:8000)
```
make serve
```
### Docker
```
docker build -t sentiment140-mlops .
docker run -p 8000:8000 sentiment140-mlops
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sentiment140_mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── sentiment140_mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sentiment140_mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

