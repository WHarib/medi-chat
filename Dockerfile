FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
        build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# cache dirs with perms
RUN mkdir -p /app/torch_cache && chmod -R 777 /app/torch_cache
ENV TORCH_HOME=/app/torch_cache XDG_CACHE_HOME=/app/torch_cache


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY densenet121_finetuned.pth thresholds.json .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
