from __future__ import annotations

import logging
from typing import Dict, List

from bil.io.schema import Segment, ScoredSegment
from bil.pipeline.ingest import InputSource
from bil.utils import image_io, optional_deps

DEFAULT_SCORING_WEIGHTS = {
    "stability": 0.4,
    "green_ratio": 0.2,
    "pitch_strip": 0.25,
    "edge_density": 0.15,
}


def _sample_video_frames(cv2, cap, start_frame: int, end_frame: int, step: int) -> List[image_io.SimpleImage]:
    frames: List = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(image_io.simple_image_from_cv2_frame(frame))
        idx += step
        if step > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return frames


def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r_f, g_f, b_f)
    min_c = min(r_f, g_f, b_f)
    delta = max_c - min_c
    if delta == 0:
        h = 0.0
    elif max_c == r_f:
        h = (60 * ((g_f - b_f) / delta) + 360) % 360
    elif max_c == g_f:
        h = 60 * ((b_f - r_f) / delta) + 120
    else:
        h = 60 * ((r_f - g_f) / delta) + 240
    s = 0.0 if max_c == 0 else delta / max_c
    v = max_c
    return h, s, v


def _green_stats(img: image_io.SimpleImage) -> tuple[float, float]:
    width = img.width
    height = img.height
    center_start = int(width * 0.3)
    center_end = int(width * 0.7)
    total_green = total_pixels = 0
    center_green = center_pixels = 0
    side_green = side_pixels = 0
    for y in range(height):
        row_start = (y * width) * 3
        for x in range(width):
            idx = row_start + x * 3
            r, g, b = img.pixels[idx : idx + 3]
            h, s, v = _rgb_to_hsv(r, g, b)
            is_green = 60 <= h <= 170 and s > 0.2 and v > 0.1
            total_pixels += 1
            if x >= center_start and x < center_end:
                center_pixels += 1
                if is_green:
                    center_green += 1
            else:
                side_pixels += 1
                if is_green:
                    side_green += 1
            if is_green:
                total_green += 1

    green_ratio = total_green / total_pixels if total_pixels else 0.0
    center_ratio = center_green / center_pixels if center_pixels else 0.0
    side_ratio = side_green / side_pixels if side_pixels else 0.0
    pitch_strip_score = max(0.0, side_ratio - center_ratio)
    return green_ratio, pitch_strip_score


def _edge_density(img: image_io.SimpleImage) -> float:
    if img.width < 2 or img.height < 2:
        return 0.0
    gray = image_io.grayscale(img)
    total = 0.0
    count = 0
    for y in range(img.height - 1):
        row = y * img.width
        next_row = (y + 1) * img.width
        for x in range(img.width - 1):
            idx = row + x
            gx = abs(gray[idx + 1] - gray[idx])
            gy = abs(gray[next_row + x] - gray[idx])
            total += gx + gy
            count += 1
    # Normalize: average gradient magnitude scaled by max possible 510 (255x2 directions)
    return float(total / (count * 510.0)) if count else 0.0


def _score_frames(
    frames: List[image_io.SimpleImage],
    seg: Segment,
    frames_fps: float,
    weights: Dict[str, float],
) -> ScoredSegment:
    if not frames:
        return ScoredSegment(seg.video, seg.start, seg.end, 0.0, "no_frames", details=None)

    gray_frames = [image_io.grayscale(f) for f in frames]
    diffs = [image_io.mean_abs_diff(gray_frames[i], gray_frames[i - 1]) for i in range(1, len(gray_frames))]
    motion = sum(diffs) / len(diffs) if diffs else 0.0
    stability_score = 1.0 / (1.0 + motion)

    reference = frames[len(frames) // 2]
    green_ratio, pitch_strip = _green_stats(reference)
    edge_density = _edge_density(reference)

    features = {
        "stability": stability_score,
        "green_ratio": green_ratio,
        "pitch_strip": pitch_strip,
        "edge_density": edge_density,
    }
    score = 0.0
    for name, value in features.items():
        score += float(weights.get(name, 0.0)) * float(value)

    reason = ",".join(f"{k}={v:.3f}" for k, v in features.items())
    details = {
        **features,
        "motion": motion,
        "frames_sampled": len(frames),
        "fps_assumed": frames_fps,
        "weights": {k: float(weights.get(k, 0.0)) for k in features.keys()},
    }
    return ScoredSegment(seg.video, seg.start, seg.end, float(score), reason, details=details)


def score_segments(
    source: InputSource, segments: List[Segment], config: dict, logger: logging.Logger
) -> List[ScoredSegment]:
    scored: List[ScoredSegment] = []
    fps_sampling = float(config.get("fps_sampling", 2))
    frames_fps = float(config.get("frames_fps", 30.0))
    weights_conf = config.get("scoring_weights") or {}
    weights: Dict[str, float] = {**DEFAULT_SCORING_WEIGHTS, **{k: float(v) for k, v in weights_conf.items()}}

    if source.kind == "video":
        if not optional_deps.has_cv2():
            logger.error("OpenCV not available for scoring video: %s", source.path)
            return [ScoredSegment(s.video, s.start, s.end, 0.0, "cv2_missing", details=None) for s in segments]
        cv2 = optional_deps.require_cv2()
        cap = cv2.VideoCapture(str(source.path))
        if not cap.isOpened():
            logger.error("Unable to open video for scoring: %s", source.path)
            return [ScoredSegment(s.video, s.start, s.end, 0.0, "decode_fail", details=None) for s in segments]

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        step = max(1, int(fps / fps_sampling) if fps_sampling > 0 else 1)

        for seg in segments:
            start_frame = int(seg.start * fps)
            end_frame = int(seg.end * fps)
            frames = _sample_video_frames(cv2, cap, start_frame, end_frame, step)
            scored.append(_score_frames(frames, seg, fps, weights))
        cap.release()
        return scored

    # frames input
    step = max(1, int(frames_fps / fps_sampling) if fps_sampling > 0 else 1)
    frame_count = len(source.frames)
    for seg in segments:
        start_idx = int(seg.start * frames_fps)
        end_idx = min(int(seg.end * frames_fps), frame_count - 1)
        frames = [source.frames[i] for i in range(start_idx, end_idx + 1, step)]
        scored.append(_score_frames(frames, seg, frames_fps, weights))
    return scored
