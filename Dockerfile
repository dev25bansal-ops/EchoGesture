# EchoGesture — reproducible runtime for predict(video_path).
#
# This image exists because the submission rules (organizer deck p.25) list
# "Docker files & other requirements" alongside the README. It packages the
# exact runtime the submitted accuracy was measured on: the nine FP16 .tflite
# members in models/ plus MediaPipe's hand_landmarker.task, run through
# src/main.py's predict().
#
#   docker build -t echogesture .
#   docker run --rm echogesture                       # smoke test on a synthetic clip
#   docker run --rm -v "$PWD/clips:/clips" echogesture /clips/sign.mp4
#
# What this image does NOT contain: the ASL Citizen corpus. It is Microsoft
# Research non-commercial data and is not redistributable, so full-split
# evaluation is not reproducible from this image alone — see docs/release/
# ARTIFACT_RIGHTS.md. The smoke test below uses a synthesised frame sequence,
# which exercises every stage of the pipeline but proves plumbing, not accuracy.

FROM python:3.11-slim-bookworm

# opencv-python links against libGL and the X stack at import time even when
# only imdecode/video decoding is used; without these `import cv2` raises
# ImportError: libGL.so.1 and the container reports a misleading "no model".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the runtime surface: the two src modules predict() imports, its local
# deps, and the shipped models. The repo's research scripts are not copied —
# they need the corpus and would only add attack surface here.
COPY src/main.py src/features.py src/landmark_frontend.py ./src/
COPY models/ ./models/
COPY app/smoke_test.py ./smoke_test.py

# Fail the build, not the run, if the ensemble or label file is missing. A
# container that starts and then reports "No model found" wastes a judge's
# time; this turns that into an immediate build error.
RUN python -c "import sys; sys.path.insert(0,'src'); import main; main._init_model(); print('model init OK:', len(main._INTERPRETERS), 'members')"

ENTRYPOINT ["python", "smoke_test.py"]
