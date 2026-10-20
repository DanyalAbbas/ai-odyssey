FROM python:3.13-slim

# insightface builds a small Cython extension at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cython3 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CPU only, offline
ENV OMP_NUM_THREADS=8
ENTRYPOINT ["python", "predict.py"]
