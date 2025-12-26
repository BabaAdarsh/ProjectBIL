from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from bil.io.schema import ScoredSegment, Track
from bil.pipeline.ingest import InputSource
from bil.utils import image_io, optional_deps


def _create_tracker():
    creator_names = [
        "TrackerCSRT_create",
        "TrackerKCF_create",
        "TrackerMOSSE_create",
    ]
    for name in creator_names:
        cv2 = optional_deps.require_cv2()
        creator = getattr(cv2, name, None)
        if creator:
            try:
                return creator()
            except Exception:
                continue
        legacy_creator = getattr(getattr(cv2, "legacy", None), name, None)
        if legacy_creator:
            try:
                return legacy_creator()
            except Exception:
                continue
    return None


def _init_bbox_from_shape(width: int, height: int, crop_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    cw, ch = crop_size
    x = max(0, (width - cw) // 2)
    y = max(0, (height - ch) // 2)
    return int(x), int(y), int(min(cw, width)), int(min(ch, height))


def track_segments(
    source: InputSource,
    segments: List[ScoredSegment],
    config: dict,
    logger: logging.Logger,
) -> List[Track]:
    tracks: List[Track] = []
    smoothing = float(config.get("smoothing", 0.8))
    crop_size_conf = config.get("crop_size", [224, 224])
    crop_size = (int(crop_size_conf[0]), int(crop_size_conf[1]))

    if source.kind == "video":
        if not optional_deps.has_cv2():
            logger.error("OpenCV not available for tracking: %s", source.path)
            return [Track(seg, boxes=[], confidence=0.0) for seg in segments]
        cv2 = optional_deps.require_cv2()
        cap = cv2.VideoCapture(str(source.path))
        if not cap.isOpened():
            logger.error("Unable to open video for tracking: %s", source.path)
            return [Track(seg, boxes=[], confidence=0.0) for seg in segments]

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        for seg in segments:
            start_frame = int(seg.start * fps)
            end_frame = int(seg.end * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if not ret:
                tracks.append(Track(seg, boxes=[], confidence=0.0))
                continue

            bbox = _init_bbox_from_shape(frame.shape[1], frame.shape[0], crop_size)
            cv_tracker = _create_tracker()
            ok = False
            if cv_tracker:
                try:
                    ok = cv_tracker.init(frame, bbox)
                except Exception:
                    ok = False

            boxes: List[List[float]] = []
            last_bbox = bbox
            current_frame = start_frame
            while current_frame <= end_frame:
                if current_frame != start_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                if cv_tracker and ok:
                    ok, tracked_bbox = cv_tracker.update(frame)
                    if ok:
                        last_bbox = tuple(map(float, tracked_bbox))
                    else:
                        ok = False
                if not ok:
                    h, w = frame.shape[:2]
                    cx = w / 2
                    cy = h / 2
                    lx, ly, lw, lh = last_bbox
                    nx = smoothing * lx + (1 - smoothing) * max(0.0, cx - lw / 2)
                    ny = smoothing * ly + (1 - smoothing) * max(0.0, cy - lh / 2)
                    last_bbox = (nx, ny, lw, lh)
                boxes.append([float(last_bbox[0]), float(last_bbox[1]), float(last_bbox[2]), float(last_bbox[3])])
                current_frame += 1

            confidence = 1.0 if ok else 0.5 if boxes else 0.0
            tracks.append(Track(seg, boxes=boxes, confidence=confidence))
        cap.release()
        return tracks

    # frames input: center crop fallback
    frame_width = source.frames[0].width if source.frames else crop_size[0]
    frame_height = source.frames[0].height if source.frames else crop_size[1]
    bbox = _init_bbox_from_shape(frame_width, frame_height, crop_size)
    frames_fps = float(source.fps or config.get("frames_fps", 30.0))
    for seg in segments:
        frames_needed = max(1, int(round((seg.end - seg.start) * frames_fps)))
        boxes = []
        for _ in range(frames_needed):
            boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
        tracks.append(Track(seg, boxes=boxes, confidence=0.5 if boxes else 0.0))
    return tracks
