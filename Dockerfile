FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.sentiment140_mlops.models.predict_model:app", "--host", "0.0.0.0", "--port", "8000"]
