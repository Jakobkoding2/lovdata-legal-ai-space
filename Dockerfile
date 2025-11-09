FROM python:3.11

WORKDIR /app

# Tools and libs many PyPI wheels expect
RUN apt-get update && apt-get install -y \
    git ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Stable pip toolchain
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# If you need PyTorch, force CPU wheels (skip CUDA):
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt || true && \
    pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY app.py .

ENV PORT=7860
CMD ["python","app.py"]
