FROM python:3.10-slim

# Install system tools + MLflow
RUN apt-get update && apt-get install -y wget unzip curl && \
    pip install mlflow

# Set working directory
WORKDIR /app

# Download and unzip your zipped mlruns structure
RUN wget https://www.leonshpaner.com/files/mlruns.zip && \
    unzip mlruns.zip && \
    rm mlruns.zip

# Move the actual experiment from models/0 to mlruns/0
RUN mkdir -p mlruns/0 && \
    cp -r mlruns/models/0/* mlruns/0/

EXPOSE 10000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "10000"]
