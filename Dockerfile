# Root evaluator image.
#
# The organizer's package layout (deck p.12) expects a Docker setup at the
# repository root; scripts/release/build_bundle.py hard-fails when this file is
# absent or empty. Python 3.11 as required on p.12.

FROM python:3.11-slim

# opencv-python and mediapipe link against X11/GL shared objects that
# python:3.11-slim does not ship. Without these the container built fine but
# every invocation died importing cv2 with
#   ImportError: libxcb.so.1: cannot open shared object file
# so `--help` never printed the contract. libEGL/libGLESv2 are loaded lazily by
# mediapipe's C++ bindings, so they only surface during a real predict() call.
# Set established by running --help AND a real prediction in the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY src/requirements.txt /app/src/requirements.txt
RUN pip install --no-cache-dir -r /app/src/requirements.txt

COPY src/ /app/src/
COPY scripts/ml/ /app/scripts/ml/
COPY results/ /app/results/

# Model weights are not redistributable and are not baked into the public
# image. Mount them at runtime:
#   docker run --rm -v /authorized/models:/app/models \
#              -v /authorized/clips:/clips echogesture /clips/sample.mp4
# predict() resolves weights as <repo>/models, i.e. /app/models inside the
# image; there is no env override, and ECHOGESTURE_MODELS was previously set
# here without any reader, which implied an override that did not exist.
#
# Two modes, both explicit:
#   no args / --help  -> prints the predict() contract and exits
#   <video> [clip_id] -> prints the organizer prediction object for that clip
# There is no subcommand CLI; the evaluator contract is predict(video_path).
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]