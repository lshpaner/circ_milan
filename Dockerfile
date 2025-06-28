FROM python:3.10-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y wget unzip && \
    pip install mlflow

# Set working directory
WORKDIR /app

# Download and unzip mlruns from your site
RUN wget https://www.leonshpaner.com/files/mlruns.zip && \
    unzip mlruns.zip && \
    rm mlruns.zip

# Expose the port MLflow uses
EXPOSE 10000

# Start MLflow server pointing to restored mlruns
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "10000"]
