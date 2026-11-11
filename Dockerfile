# Root evaluator image.
#
# The organizer's package layout (deck p.12) expects a Docker setup at the
# repository root; scripts/release/build_bundle.py hard-fails when this file is
# absent or empty. Python 3.11 as required on p.12.

FROM python:3.11-slim

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
# Without them predict() raises FileNotFoundError naming the missing member
# rather than running fewer model than the metadata declares.
ENV ECHOGESTURE_MODELS=/app/models

# Two modes, both explicit:
#   no args / --help  -> prints the predict() contract and exits
#   <video> [clip_id] -> prints the organizer prediction object for that clip
# There is no subcommand CLI; the evaluator contract is predict(video_path).
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]