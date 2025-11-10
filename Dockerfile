# Minimal Dockerfile tuned for Render web services (HF API-only, no local ML libs)
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN chmod +x /app/start.sh

# Render will inject a PORT environment variable that the app listens on.
# BACKEND_URL should be provided at runtime to point at the /ask_law backend API.
EXPOSE 7860
CMD ["./start.sh"]
