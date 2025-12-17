from __future__ import annotations

import logging
from typing import List

from bil.io.schema import Segment, ScoredSegment
from bil.pipeline.ingest import InputSource
from bil.utils import image_io, optional_deps


def _sample_video_frames(cv2, cap, start_frame: int, end_frame: int, step: int) -> List:
    frames: List = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += step
        if step > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return frames


def _score_frames(frames: List[image_io.SimpleImage], fps_sampling: float, seg: Segment) -> ScoredSegment:
    if not frames:
        return ScoredSegment(seg.video, seg.start, seg.end, 0.0, "no_frames")
    gray_frames = [image_io.grayscale(f) for f in frames]
    diffs = [image_io.mean_abs_diff(gray_frames[i], gray_frames[i - 1]) for i in range(1, len(gray_frames))]
    stability = sum(diffs) / len(diffs) if diffs else 0.0
    color_std = image_io.average_std(frames)
    stability_score = 1.0 / (1.0 + stability)
    texture_score = color_std / (color_std + 50.0)
    score = float(0.6 * stability_score + 0.4 * texture_score)
    reason = f"stability={stability:.2f},texture={color_std:.2f}"
    return ScoredSegment(seg.video, seg.start, seg.end, score, reason)


def score_segments(
    source: InputSource, segments: List[Segment], config: dict, logger: logging.Logger
) -> List[ScoredSegment]:
    scored: List[ScoredSegment] = []
    fps_sampling = float(config.get("fps_sampling", 2))

    if source.kind == "video":
        if not optional_deps.cv2_available():
            logger.error("OpenCV not available for scoring video: %s", source.path)
            return [ScoredSegment(s.video, s.start, s.end, 0.0, "cv2_missing") for s in segments]
        cv2 = optional_deps.require_cv2()
        cap = cv2.VideoCapture(str(source.path))
        if not cap.isOpened():
            logger.error("Unable to open video for scoring: %s", source.path)
            return [ScoredSegment(s.video, s.start, s.end, 0.0, "decode_fail") for s in segments]

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        step = max(1, int(fps / fps_sampling) if fps_sampling > 0 else 1)

        for seg in segments:
            start_frame = int(seg.start * fps)
            end_frame = int(seg.end * fps)
            frames = _sample_video_frames(cv2, cap, start_frame, end_frame, step)
            if not frames:
                scored.append(ScoredSegment(seg.video, seg.start, seg.end, 0.0, "no_frames"))
                continue
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            diffs = [float(cv2.absdiff(gray_frames[i], gray_frames[i - 1]).mean()) for i in range(1, len(gray_frames))]
            stability = sum(diffs) / len(diffs) if diffs else 0.0
            color_std = float(sum(f.std() for f in frames) / len(frames))
            stability_score = 1.0 / (1.0 + stability)
            texture_score = color_std / (color_std + 50.0)
            score = float(0.6 * stability_score + 0.4 * texture_score)
            reason = f"stability={stability:.2f},texture={color_std:.2f}"
            scored.append(ScoredSegment(seg.video, seg.start, seg.end, score, reason))
        cap.release()
        return scored

    # frames input
    step = max(1, int(15 / fps_sampling) if fps_sampling > 0 else 1)
    frame_count = len(source.frames)
    for seg in segments:
        start_idx = int(seg.start * 15)
        end_idx = min(int(seg.end * 15), frame_count - 1)
        frames = [source.frames[i] for i in range(start_idx, end_idx + 1, step)]
        scored.append(_score_frames(frames, fps_sampling, seg))
    return scored
