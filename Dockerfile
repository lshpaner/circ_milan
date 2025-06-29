FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . .

# Set environment variables (optional for MLflow)
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expose port
EXPOSE 5000

# Launch MLflow tracking server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:/app/mlruns"]
