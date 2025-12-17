from __future__ import annotations


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        raise ImportError("OpenCV not installed. Install with: pip install opencv-python") from exc
    return cv2


def cv2_available() -> bool:
    try:
        import cv2  # type: ignore

        return True
    except Exception:
        return False
