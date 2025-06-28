# Dockerfile
FROM python:3.10-slim

# Install MLflow
RUN pip install mlflow

# Create working directory
WORKDIR /app

# Expose port used by Render
EXPOSE 10000

# Run MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "./mlruns", \
     "--host", "0.0.0.0", "--port", "10000"]

