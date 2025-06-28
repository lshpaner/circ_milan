FROM python:3.10-slim

# Install required system tools and MLflow
RUN apt-get update && apt-get install -y wget unzip curl && \
    pip install mlflow

# Set working directory
WORKDIR /app

# Download and extract the zipped mlruns folder
RUN wget https://www.leonshpaner.com/files/mlruns.zip && \
    unzip mlruns.zip && \
    rm mlruns.zip

# Expose MLflow port
EXPOSE 10000

# Start MLflow server using the unzipped mlruns directory
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "10000"]
