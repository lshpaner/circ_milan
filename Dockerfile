# Use a slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (to take advantage of Docker caching)
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose port
EXPOSE 5000

# Run the MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "file:/app/mlruns", "--default-artifact-root", "file:/app/mlruns", "--host", "0.0.0.0", "--port", "5000",]
