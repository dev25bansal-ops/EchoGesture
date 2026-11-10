"""EchoGesture demo lab — camera-less, real-prediction HUD smoke runs.

Module entry point: ``python -m app.demo_lab --video <clip.mp4>``.
Harness self-checks: ``python -m app.demo_lab --check``.
See ``__init__.py`` for the full contract. No fake captions are generated;
every output comes from the repository's real ``predict(video_path)``.
"""

from . import check, run_video, main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["check", "run_video", "main"]
