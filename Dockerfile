FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y wget unzip curl && \
    pip install mlflow

# Set working directory
WORKDIR /app

# Download and unzip your mlruns zip file
RUN wget https://www.leonshpaner.com/files/mlruns.zip && \
    unzip mlruns.zip && \
    rm mlruns.zip

# Optional: move your experiment folder to mlruns/0 so MLflow UI shows it
RUN mkdir -p mlruns/0 && \
    cp -r mlruns/models/452642104975561062/* mlruns/0/

EXPOSE 10000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "10000"]
