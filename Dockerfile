# Use python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for compiling C extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run injects the PORT variable (usually 8080)
# We default to 8080 just in case
ENV PORT=8080

# CRITICAL CHANGE: 
# 1. We use "exec" (better signal handling)
# 2. We bind to :$PORT (Dynamic) instead of 10000
CMD exec gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind :$PORT --timeout 0
