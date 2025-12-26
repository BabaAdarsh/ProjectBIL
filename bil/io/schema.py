from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any


class FailureCode(str, Enum):
    VIDEO_DECODE_FAIL = "VIDEO_DECODE_FAIL"
    NO_SEGMENTS = "NO_SEGMENTS"
    LOW_SCORE = "LOW_SCORE"
    TRACK_FAIL = "TRACK_FAIL"
    CLIP_WRITE_FAIL = "CLIP_WRITE_FAIL"


@dataclass
class Segment:
    video: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredSegment(Segment):
    score: float
    reason: str
    details: Dict[str, Any] | None = None


@dataclass
class Track:
    segment: ScoredSegment
    boxes: List[List[float]]  # [x, y, w, h] per frame
    confidence: float


def serialize_tracks(tracks: List[Track]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for track in tracks:
        seg = track.segment.to_dict()
        seg.update(
            {
                "score": track.segment.score,
                "reason": track.segment.reason,
                "boxes": track.boxes,
                "confidence": track.confidence,
            }
        )
        serialized.append(seg)
    return serialized


def ensure_seconds(value: float) -> float:
    return max(0.0, float(value))
