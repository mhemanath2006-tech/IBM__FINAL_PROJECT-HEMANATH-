FROM python:3.11-slim

# insightface is compiled from source, so it needs a C/C++ toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir "numpy<2" cython setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Download the pretrained face model at build time so the first scan is not slow.
RUN python -c "from insightface.app import FaceAnalysis; a = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=['detection','recognition']); a.prepare(ctx_id=-1, det_size=(480,480)); print('model ready')"

COPY --chown=user . .

EXPOSE 7860
# One worker keeps memory low on the free tier; threads handle concurrent students.
CMD ["gunicorn", "wsgi:app", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120"]
