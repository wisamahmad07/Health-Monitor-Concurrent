# Use official Python 3.10.12 image
FROM python:3.10.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    build-essential \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

# Install pip packages including PyTorch CPU versions
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--timeout", "60000", "--workers", "1", "--worker-class", "gevent", "--worker-connections", "100", "main:app"]
