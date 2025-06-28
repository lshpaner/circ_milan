FROM python:3.10-slim

RUN apt-get update && apt-get install -y wget unzip curl && \
    pip install mlflow

WORKDIR /app

# Download and extract the zipped mlruns folder
RUN wget https://www.leonshpaner.com/files/mlruns.zip && \
    unzip mlruns.zip && \
    rm mlruns.zip

# Move models/0 → mlruns/0 (exact experiment ID placement)
RUN rm -rf mlruns/0 && \
    mv mlruns/models/0 mlruns/0

EXPOSE 10000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "10000"]
