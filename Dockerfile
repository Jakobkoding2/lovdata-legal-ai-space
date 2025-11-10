# Minimal Dockerfile tuned for Render web services
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal system deps (kept small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only what is needed for pip install first (speeds rebuilds)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the source
COPY . /app

# Ensure start script is executable
RUN chmod +x /app/start.sh

# Render will inject a PORT environment variable that the app listens on.
EXPOSE 8080

# Default start command for the container-based service on Render
CMD ["./start.sh"]
