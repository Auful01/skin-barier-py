FROM python:3.10-slim

# Install system deps untuk Pillow dll
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Install python deps
RUN pip install --no-cache-dir flask flask-cors tensorflow numpy pillow

# Copy semua file project
COPY . .

# Expose Flask port
EXPOSE 5000

# Jalankan Flask
CMD ["python", "app.py"]