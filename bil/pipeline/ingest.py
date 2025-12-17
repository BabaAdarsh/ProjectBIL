from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from bil.utils import image_io

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class InputSource:
    path: Path
    kind: str  # "video" or "frames"
    frames: List[image_io.SimpleImage]


def collect_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    videos: List[Path] = []
    for ext in VIDEO_EXTS:
        videos.extend(path.rglob(f"*{ext}"))
    return sorted(videos)


def _read_frames_in_dir(path: Path) -> List[InputSource]:
    image_files = sorted([p for p in path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_files:
        return []
    frames = []
    for p in image_files:
        frames.append(image_io.read_image(p))
    return [InputSource(path=path, kind="frames", frames=frames)]


def collect_sources(path: Path, input_format: str = "auto") -> List[InputSource]:
    sources: List[InputSource] = []
    fmt = input_format.lower()

    if fmt == "frames":
        if path.is_dir():
            return _read_frames_in_dir(path)
        return []

    if fmt == "video":
        if path.is_file():
            return [InputSource(path=path, kind="video", frames=[])]
        for video in collect_inputs(path):
            sources.append(InputSource(path=video, kind="video", frames=[]))
        return sources

    # auto detection
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [InputSource(path=path, kind="video", frames=[])]
    if path.is_dir():
        frames_sources = _read_frames_in_dir(path)
        if frames_sources:
            return frames_sources
        for video in collect_inputs(path):
            sources.append(InputSource(path=video, kind="video", frames=[]))
    return sources
