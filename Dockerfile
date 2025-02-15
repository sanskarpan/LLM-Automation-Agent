FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set SSL certificate path (fixes SSL issues)
ENV SSL_CERT_FILE=/usr/local/lib/python3.10/site-packages/certifi/cacert.pem

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt certifi

# Copy the rest of the application
COPY . .

# Create and set permissions for data directory
RUN mkdir -p /data && chmod -R 777 /data

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]