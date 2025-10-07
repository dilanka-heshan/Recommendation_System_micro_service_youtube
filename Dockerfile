# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variables for model handling and BGE compatibility
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache
ENV TOKENIZERS_PARALLELISM=false
ENV MODEL_CACHE_DIR=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_CPP_MIN_LOG_LEVEL=3

# Pre-download and bundle models into the Docker image for faster startup
# Create necessary directories and download models with proper error handling
RUN mkdir -p /app/models /app/.cache && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 --retries=3 torch sentence-transformers transformers tokenizers sentencepiece protobuf

# Copy the model download script
COPY download_models.py /tmp/download_models.py

# Run the model download script with timeout
RUN timeout 1800 python /tmp/download_models.py || echo "Model bundling failed, continuing build..."

# Copy the entire application
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose the port that the app runs on
EXPOSE 8080

# Set startup timeout to handle model loading
ENV STARTUP_TIMEOUT=600

# Command to run the application - hardcoded to port 8080  
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "300"]
