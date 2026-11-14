FROM python:3.11-slim

# Models are downloaded here on first load; mount a volume so they survive restarts.
ENV HF_HUB_CACHE=/models/huggingface \
    PYTHONUNBUFFERED=1

# ffmpeg converts uploaded audio to 16kHz mono WAV (torchcodec also loads its libraries).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# The server has no GPU: take the torch family from the CPU-only index before anything can
# pull the default build. torchaudio must come from this index too: the PyPI Linux ARM64
# wheel (2.11.1) is linked against CUDA and fails to import on CPU torch.
# This layer changes rarely, so it stays cached when only app code changes.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchaudio==2.11.0 torchcodec==0.17.0 \
    && pip install --no-cache-dir pyannote.audio==4.0.7

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir ".[voice-pyannote]"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
