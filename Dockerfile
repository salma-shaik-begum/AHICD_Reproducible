# AHICF Reproduction Environment
# CPU-only Docker image for reproducibility

FROM python:3.11-slim

# System libraries required by OpenCV and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline source code
COPY src/ ./src/

# Runtime configuration
ENV AHICF_BASE_DIR=/app/data
ENV AHICF_DATASET_DIR=/app/data/Dataset
ENV PYTHONUNBUFFERED=1

# Execute the pipeline from the source directory
WORKDIR /app/src

# Default command: run the core AHICF pipeline
ENTRYPOINT ["python", "main.py"]
CMD ["--stages", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
