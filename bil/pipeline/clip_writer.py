from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from bil.io.schema import FailureCode, Track
from bil.pipeline.ingest import InputSource
from bil.utils import image_io, optional_deps


def _build_writer(cv2, path: Path, fps: float, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, frame_size)


def _simpleimage_from_cv2_frame(frame) -> image_io.SimpleImage:
    height, width = frame.shape[:2]
    # cv2 uses BGR, flatten to RGB
    pixels: List[int] = []
    for y in range(height):
        for x in range(width):
            b, g, r = frame[y, x]
            pixels.extend([int(r), int(g), int(b)])
    return image_io.SimpleImage(width=width, height=height, pixels=pixels)


def _write_frames_output(
    clips_dir: Path,
    track: Track,
    frames: List[image_io.SimpleImage],
    crop_size: List[int],
) -> (Dict, int):
    crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
    clip_subdir = clips_dir / f"clip_{len(list(clips_dir.glob('clip_*'))):04d}"
    frame_paths = []
    for idx, box in enumerate(track.boxes[: len(frames)]):
        x, y, w, h = map(int, box)
        img = frames[idx]
        cropped = image_io.crop(img, x, y, w, h)
        resized = image_io.resize_nearest(cropped, crop_w, crop_h)
        frame_path = clip_subdir / f"frame_{idx:06d}.png"
        image_io.write_image(frame_path, resized)
        frame_paths.append(frame_path)
    return (
        {
            "video": track.segment.video,
            "start": track.segment.start,
            "end": track.segment.end,
            "score": track.segment.score,
            "path": str(clip_subdir),
            "failure": None if frame_paths else FailureCode.CLIP_WRITE_FAIL.value,
        },
        len(frame_paths),
    )


def write_clips(
    source: InputSource,
    tracks: List[Track],
    config: dict,
    run_dir: Path,
    logger: logging.Logger,
    output_format: str,
) -> List[Dict]:
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    requested_format = output_format
    if requested_format == "auto":
        requested_format = "mp4" if source.kind == "video" and optional_deps.has_cv2() else "frames"

    crop_size_conf = config.get("crop_size", [224, 224])
    metadata: List[Dict] = []

    if requested_format == "frames":
        fps = 15.0 if source.kind == "frames" else 24.0
        all_frames: List[image_io.SimpleImage] = []
        if source.kind == "frames":
            all_frames = source.frames
        elif optional_deps.has_cv2():
            cv2 = optional_deps.require_cv2()
            cap = cv2.VideoCapture(str(source.path))
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(_simpleimage_from_cv2_frame(frame))
                cap.release()
        for track in tracks:
            start_idx = int(track.segment.start * fps)
            end_idx = min(int(track.segment.end * fps), start_idx + len(track.boxes))
            frames_slice = all_frames[start_idx:end_idx]
            meta, written = _write_frames_output(clips_dir, track, frames_slice, crop_size_conf)
            if written == 0:
                meta["failure"] = FailureCode.CLIP_WRITE_FAIL.value
                meta["path"] = None
            metadata.append(meta)
        return metadata

    # mp4 output
    if not optional_deps.has_cv2():
        logger.error("OpenCV not available for mp4 output, falling back to frames.")
        return write_clips(source, tracks, config, run_dir, logger, output_format="frames")
    cv2 = optional_deps.require_cv2()

    target_fps = float(config.get("output_fps", 24.0))
    crop_w, crop_h = int(crop_size_conf[0]), int(crop_size_conf[1])

    for idx, track in enumerate(tracks):
        if source.kind == "frames":
            fps = 15.0
        else:
            cap = cv2.VideoCapture(str(source.path))
            if not cap.isOpened():
                metadata.append(
                    {
                        "video": track.segment.video,
                        "start": track.segment.start,
                        "end": track.segment.end,
                        "score": track.segment.score,
                        "path": None,
                        "failure": FailureCode.CLIP_WRITE_FAIL.value,
                    }
                )
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        start_frame = int(track.segment.start * fps)
        end_frame = min(int(track.segment.end * fps), start_frame + len(track.boxes))
        if end_frame <= start_frame:
            metadata.append(
                {
                    "video": track.segment.video,
                    "start": track.segment.start,
                    "end": track.segment.end,
                    "score": track.segment.score,
                    "path": None,
                    "failure": FailureCode.CLIP_WRITE_FAIL.value,
                }
            )
            if source.kind == "video":
                cap.release()
            continue

        clip_path = clips_dir / f"clip_{idx:04d}.mp4"
        writer = _build_writer(cv2, clip_path, target_fps, (crop_w, crop_h))
        frames_written = 0

        if source.kind == "frames":
            frames = source.frames[start_frame:end_frame]
            for local_idx, box in enumerate(track.boxes[: len(frames)]):
                frame = frames[local_idx]
                x, y, w, h = map(int, box)
                cropped = image_io.crop(frame, x, y, w, h)
                resized = image_io.resize_nearest(cropped, crop_w, crop_h)
                # convert to BGR numpy array
                import numpy as np

                arr = np.array(resized.pixels, dtype=np.uint8).reshape(resized.height, resized.width, 3)
                bgr = arr[:, :, ::-1]
                writer.write(bgr)
                frames_written += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for box in track.boxes[: end_frame - start_frame]:
                ret, frame = cap.read()
                if not ret:
                    break
                x, y, w, h = map(int, box)
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                cropped = frame[y : y + h, x : x + w]
                resized = cv2.resize(cropped, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                writer.write(resized)
                frames_written += 1

        writer.release()
        if source.kind == "video":
            cap.release()

        if frames_written == 0:
            clip_path.unlink(missing_ok=True)
            metadata.append(
                {
                    "video": track.segment.video,
                    "start": track.segment.start,
                    "end": track.segment.end,
                    "score": track.segment.score,
                    "path": None,
                    "failure": FailureCode.CLIP_WRITE_FAIL.value,
                }
            )
        else:
            metadata.append(
                {
                    "video": track.segment.video,
                    "start": track.segment.start,
                    "end": track.segment.end,
                    "score": track.segment.score,
                    "path": str(clip_path),
                    "failure": None,
                }
            )
    return metadata
