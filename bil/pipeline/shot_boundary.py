from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from bil.io.schema import Segment, ensure_seconds
from bil.pipeline.ingest import InputSource
from bil.utils import image_io, optional_deps


def _detect_with_pyscenedetect(video_path: Path, min_len: float, max_len: float) -> List[Segment]:
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception:
        return []

    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate().get_fps() if video_manager.get_framerate() else 24.0
    segments: List[Segment] = []
    for start, end in scene_list:
        start_sec = start.get_frames() / fps
        end_sec = end.get_frames() / fps
        if end_sec - start_sec < min_len:
            continue
        length = end_sec - start_sec
        while length > max_len:
            s = start_sec
            e = s + max_len
            segments.append(Segment(str(video_path), ensure_seconds(s), ensure_seconds(e)))
            start_sec = e
            length = end_sec - start_sec
        if end_sec - start_sec >= min_len:
            segments.append(Segment(str(video_path), ensure_seconds(start_sec), ensure_seconds(end_sec)))
    video_manager.release()
    return segments


def _detect_from_frames(
    source: InputSource, min_len: float, max_len: float, fps_sampling: float, diff_threshold: float, logger: logging.Logger
) -> List[Segment]:
    frames = source.frames
    if not frames:
        return []
    fps = float(source.frames and 15.0)
    sample_interval = max(1, int(round(fps / fps_sampling))) if fps_sampling > 0 else 1
    boundaries = [0]
    prev_gray = None
    for idx, frame in enumerate(frames):
        if idx % sample_interval != 0:
            continue
        gray = image_io.grayscale(frame)
        if prev_gray is not None:
            score = image_io.mean_abs_diff(gray, prev_gray)
            if score > diff_threshold:
                boundaries.append(idx)
        prev_gray = gray
    boundaries.append(len(frames))
    segments: List[Segment] = []
    for start_frame, end_frame in zip(boundaries[:-1], boundaries[1:]):
        start_sec = ensure_seconds(start_frame / fps)
        end_sec = ensure_seconds(end_frame / fps)
        length = end_sec - start_sec
        if length < min_len:
            continue
        while length > max_len:
            segments.append(Segment(str(source.path), start_sec, start_sec + max_len))
            start_sec += max_len
            length = end_sec - start_sec
        if end_sec - start_sec >= min_len:
            segments.append(Segment(str(source.path), start_sec, end_sec))
    logger.info("Frame-diff detector (frames input) used for %s, found %d segments", source.path, len(segments))
    return segments


def detect_shots(
    source: InputSource,
    config: dict,
    logger: logging.Logger,
) -> Tuple[List[Segment], bool]:
    min_len = float(config.get("min_segment_len", 1.0))
    max_len = float(config.get("max_segment_len", 10.0))
    fps_sampling = float(config.get("fps_sampling", 2))
    diff_threshold = float(config.get("diff_threshold", 25.0))

    if source.kind == "video":
        segments = _detect_with_pyscenedetect(source.path, min_len, max_len)
        if segments:
            logger.info("PySceneDetect used for %s, found %d segments", source.path, len(segments))
            return segments, True
        if not optional_deps.cv2_available():
            logger.error("Unable to open video without OpenCV: %s", source.path)
            return [], False
        cv2 = optional_deps.require_cv2()
        cap = cv2.VideoCapture(str(source.path))
        if not cap.isOpened():
            logger.error("Unable to open video: %s", source.path)
            return [], False

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_interval = max(1, int(round(fps / fps_sampling))) if fps_sampling > 0 else 1

        boundaries = [0]
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(diff.mean())
                if score > diff_threshold:
                    boundaries.append(frame_idx)
            prev_gray = gray
            frame_idx += 1

        boundaries.append(frame_count)
        cap.release()

        segments_out: List[Segment] = []
        for start_frame, end_frame in zip(boundaries[:-1], boundaries[1:]):
            start_sec = ensure_seconds(start_frame / fps)
            end_sec = ensure_seconds(end_frame / fps)
            length = end_sec - start_sec
            if length < min_len:
                continue
            while length > max_len:
                segments_out.append(Segment(str(source.path), start_sec, start_sec + max_len))
                start_sec += max_len
                length = end_sec - start_sec
            if end_sec - start_sec >= min_len:
                segments_out.append(Segment(str(source.path), start_sec, end_sec))

        logger.info("Frame-diff detector used for %s, found %d segments", source.path, len(segments_out))
        return segments_out, True

    # frames input
    segments = _detect_from_frames(source, min_len, max_len, fps_sampling, diff_threshold, logger)
    return segments, True
