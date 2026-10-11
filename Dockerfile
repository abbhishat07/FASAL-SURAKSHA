# ============================================================
# FASAL SURAKSHA (फसल सुरक्षा) - Production Dockerfile
# Deployment ready for Hugging Face Spaces, Render, AWS ECS, or GCP
# ============================================================

FROM python:3.11-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install essential system dependencies for OpenCV, PyTorch, and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree and model assets
COPY . /app

# Ensure directories exist
RUN mkdir -p dataset models utils reports_cache

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch both FastAPI (background) and Streamlit (foreground)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false"]
