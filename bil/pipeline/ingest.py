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


def collect_sources(path: Path) -> List[InputSource]:
    if path.is_file():
        return [InputSource(path=path, kind="video", frames=[])]
    frame_paths = sorted([p for p in path.glob("*.png") if p.is_file()])
    if frame_paths:
        frames = [image_io.read_image(p) for p in frame_paths]
        return [InputSource(path=path, kind="frames", frames=frames)]
    sources: List[InputSource] = []
    for video in collect_inputs(path):
        sources.append(InputSource(path=video, kind="video", frames=[]))
    return sources
