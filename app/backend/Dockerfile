FROM python:3.9-slim

WORKDIR /app

# Install curl (needed for health check)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Ensure model directory exists
RUN mkdir -p /app/model

# Copy only the requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY main.py .

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]