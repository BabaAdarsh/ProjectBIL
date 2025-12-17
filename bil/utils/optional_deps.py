from __future__ import annotations


def has_cv2() -> bool:
    try:
        import cv2  # type: ignore

        return True
    except Exception:
        return False


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        raise ImportError("OpenCV not installed. Install: pip install opencv-python") from exc
    return cv2
