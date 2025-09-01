FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/main

# Create non-root user
RUN useradd -m -u 1000 mluser

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=mluser:mluser . .

# Set up volumes and environment
VOLUME ["/data"]
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER mluser

CMD ["python", "./main.py"]