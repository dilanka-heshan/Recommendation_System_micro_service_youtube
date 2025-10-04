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
    pip install --no-cache-dir --timeout=1000 --retries=3 torch sentence-transformers transformers tokenizers && \
    timeout 1800 python -c "import os; os.environ['TOKENIZERS_PARALLELISM']='false'; \
    import sys; \
    try: \
        from sentence_transformers import SentenceTransformer, CrossEncoder; \
        import warnings; warnings.filterwarnings('ignore'); \
        print('=' * 60); \
        print('Starting model bundling process...'); \
        print('=' * 60); \
        print('Downloading BAAI/bge-base-en embedding model...'); \
        model1 = SentenceTransformer('BAAI/bge-base-en', trust_remote_code=True, device='cpu'); \
        print('Saving BAAI/bge-base-en to /app/models/bge-base-en...'); \
        model1.save('/app/models/bge-base-en'); \
        print('✅ BAAI/bge-base-en model bundled successfully'); \
        print('-' * 60); \
        print('Downloading BAAI/bge-reranker-base model...'); \
        model2 = CrossEncoder('BAAI/bge-reranker-base', trust_remote_code=True, device='cpu'); \
        print('Saving BAAI/bge-reranker-base to /app/models/bge-reranker-base...'); \
        model2.save('/app/models/bge-reranker-base'); \
        print('✅ BAAI/bge-reranker-base model bundled successfully'); \
        print('-' * 60); \
        print('Verifying bundled models...'); \
        assert os.path.exists('/app/models/bge-base-en'), '❌ Embedding model directory not found'; \
        assert os.path.isdir('/app/models/bge-base-en'), '❌ Embedding model path is not a directory'; \
        assert os.path.exists('/app/models/bge-reranker-base'), '❌ Reranker model directory not found'; \
        assert os.path.isdir('/app/models/bge-reranker-base'), '❌ Reranker model path is not a directory'; \
        print(f'Embedding model contents: {os.listdir(\"/app/models/bge-base-en\")}'); \
        print(f'Reranker model contents: {os.listdir(\"/app/models/bge-reranker-base\")}'); \
        print('=' * 60); \
        print('✅ All models verified successfully'); \
        print('=' * 60); \
    except Exception as e: \
        print('=' * 60); \
        print(f'❌ Model bundling failed: {e}'); \
        print('Models will be downloaded at runtime from HuggingFace'); \
        print('=' * 60); \
        import traceback; traceback.print_exc()" || echo "Model bundling failed, continuing build..."

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
