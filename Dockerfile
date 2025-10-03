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

# Pre-download and bundle models into the Docker image for faster startup
# This is optional - if it fails, the app will fallback to downloading at runtime
RUN mkdir -p /app/models && \
    python -c "import sys; \
    try: \
        from sentence_transformers import SentenceTransformer, CrossEncoder; \
        print('Downloading and bundling BAAI/bge-base-en embedding model...'); \
        SentenceTransformer('BAAI/bge-base-en').save('/app/models/bge-base-en'); \
        print('BAAI/bge-base-en model bundled successfully'); \
        print('Downloading and bundling BAAI/bge-reranker-base model...'); \
        CrossEncoder('BAAI/bge-reranker-base').save('/app/models/bge-reranker-base'); \
        print('BAAI/bge-reranker-base model bundled successfully'); \
    except Exception as e: \
        print(f'Model bundling failed: {e}. Models will be downloaded at runtime.'); \
        sys.exit(0)" || echo "Model bundling failed, continuing build..."

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
