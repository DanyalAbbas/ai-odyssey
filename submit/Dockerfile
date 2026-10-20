FROM python:3.13-slim

# insightface compiles a small Cython extension at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cython3 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY . .

# CPU only, offline at run time
ENV OMP_NUM_THREADS=8
ENTRYPOINT ["python", "predict.py"]
