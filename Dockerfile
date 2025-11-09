FROM python:3.11-slim

# Work directory
WORKDIR /app

# Basic system tools
RUN apt-get update && apt-get install -y \
    build-essential git curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip + build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Copy and install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy app code only
COPY app.py .

# Environment and start
ENV PORT=7860
CMD ["python", "app.py"]
