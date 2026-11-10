"""Module entry point: ``python -m app.v7_demo --video <clip.mp4>``."""

from .cli import main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
