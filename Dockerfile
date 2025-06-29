FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and mlruns folder
COPY . .

# Expose MLflow UI port
EXPOSE 5000

# Launch MLflow UI pointing to your local mlruns
CMD ["mlflow", "server", \
     "--backend-store-uri", "file:/app/mlruns", \
     "--default-artifact-root", "file:/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
