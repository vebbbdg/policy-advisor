FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# CPU-only torch: the default wheel bundles ~2GB of CUDA libs we never use on
# Render's CPU instances; the CPU wheel keeps the image lean.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Cap BLAS threads: the free 512MB instance can't afford many worker threads.
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HOME=/app/.hf
# Pre-bake the ONNX embedding model (all-MiniLM-L6-v2 via onnxruntime) into the
# image so cold starts load it offline - no chroma S3 call at boot. The default
# EMBEDDING_BACKEND=onnx never imports torch, keeping the free tier within 512MB.
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['bake'])"

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/vectordb data/uploads logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
