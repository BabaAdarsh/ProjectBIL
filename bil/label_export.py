from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from bil.io.json_utils import load_json
from bil.utils import optional_deps

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
NUM_KEYFRAMES = 4


def _require_pillow():
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard only
        raise ImportError("Pillow is required for label export. Install: pip install pillow") from exc
    return Image


def _load_segments(run_dir: Path) -> List[Dict]:
    segments_path = run_dir / "segments_scored.json"
    if not segments_path.exists():
        raise FileNotFoundError(f"segments_scored.json not found under {run_dir}")
    data = load_json(segments_path)
    if not isinstance(data, list):
        raise ValueError("segments_scored.json must contain a list of segments")
    return data


def _list_frame_files(path: Path) -> List[Path]:
    return sorted([p for p in path.glob("*") if p.suffix.lower() in IMAGE_EXTS])


def _select_indices(start: float, end: float, fps: float, total_frames: int, num_samples: int = NUM_KEYFRAMES) -> List[int]:
    if fps <= 0 or total_frames <= 0:
        return []
    start_idx = max(0, int(start * fps))
    end_idx = min(total_frames - 1, int(end * fps)) if total_frames else start_idx
    if end_idx < start_idx:
        end_idx = start_idx
    if num_samples <= 1 or end_idx == start_idx:
        return [start_idx] * max(1, num_samples)
    span = end_idx - start_idx
    indices = []
    for i in range(num_samples):
        frac = i / (num_samples - 1)
        idx = start_idx + int(span * frac)
        indices.append(min(total_frames - 1, idx))
    return indices


def _load_frames_from_dir(
    files: List[Path], indices: List[int], ImageMod
) -> List["ImageMod.Image"]:  # type: ignore[name-defined]
    images = []
    for idx in indices:
        if idx < 0 or idx >= len(files):
            continue
        with ImageMod.open(files[idx]) as im:
            images.append(im.convert("RGB").copy())
    return images


def _load_frames_from_video(cv2, cap, indices: List[int], ImageMod) -> List["ImageMod.Image"]:  # type: ignore[name-defined]
    frames: List = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(ImageMod.fromarray(rgb))
    return frames


def _make_montage(images: List, ImageMod) -> "ImageMod.Image" | None:  # type: ignore[name-defined]
    if not images:
        return None
    base_w, base_h = images[0].size
    canvas = ImageMod.new("RGB", (base_w * 2, base_h * 2))
    for idx, img in enumerate(images[:NUM_KEYFRAMES]):
        x = (idx % 2) * base_w
        y = (idx // 2) * base_h
        canvas.paste(img.resize((base_w, base_h)), (x, y))
    return canvas


def export_labels(run_dir: Path, out_dir: Path, top_k: int) -> int:
    try:
        ImageMod = _require_pillow()
    except ImportError as exc:
        print(f"[label_export] {exc}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir = out_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    try:
        segments_raw = _load_segments(run_dir)
    except Exception as exc:
        print(f"[label_export] Failed to load segments: {exc}")
        return 1

    config_path = run_dir / "config_resolved.json"
    config_resolved: Dict = load_json(config_path) if config_path.exists() else {}
    frames_fps = float(config_resolved.get("frames_fps") or config_resolved.get("frames_fps_resolved") or 30.0)
    input_kind = config_resolved.get("input_kind_detected") or config_resolved.get("input_format_resolved")

    indexed_segments = [{"_id": idx, **seg} for idx, seg in enumerate(segments_raw)]
    sorted_segments = sorted(indexed_segments, key=lambda s: float(s.get("score", 0.0)), reverse=True)[:top_k]

    frame_cache: Dict[Path, List[Path]] = {}
    video_cache: Dict[Path, Tuple] = {}
    csv_path = out_dir / "segments.csv"

    if input_kind == "video" and not optional_deps.has_cv2():
        print("[label_export] OpenCV is required to export keyframes for video-mode runs. Install opencv-python.")
        return 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", "video", "start", "end", "score", "label", "details"])

        cv2 = optional_deps.require_cv2() if optional_deps.has_cv2() else None

        for export_idx, seg in enumerate(sorted_segments):
            video_path = Path(seg.get("video", ""))
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            score = float(seg.get("score", 0.0))
            details_data = {"reason": seg.get("reason", ""), "details": seg.get("details") or {}}
            details_str = json.dumps(details_data)

            writer.writerow([seg["_id"], str(video_path), start, end, score, "", details_str])

            if video_path.is_dir():
                if video_path not in frame_cache:
                    frame_cache[video_path] = _list_frame_files(video_path)
                files = frame_cache[video_path]
                indices = _select_indices(start, end, frames_fps, len(files))
                images = _load_frames_from_dir(files, indices, ImageMod)
            elif input_kind == "frames":
                # Frames mode but path missing: skip silently to keep CSV usable.
                images = []
            else:
                if not cv2:
                    print(f"[label_export] Skipping keyframes for {video_path}: OpenCV missing.")
                    continue
                if video_path not in video_cache:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        print(f"[label_export] Unable to open video: {video_path}")
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    video_cache[video_path] = (cap, fps, total_frames)
                cap, fps, total_frames = video_cache[video_path]
                indices = _select_indices(start, end, fps, total_frames or int((end - start) * fps))
                images = _load_frames_from_video(cv2, cap, indices, ImageMod)

            montage = _make_montage(images, ImageMod)
            if montage:
                montage_path = keyframe_dir / f"segment_{export_idx:04d}.png"
                montage.save(montage_path)

    for cap, _, _ in video_cache.values():
        cap.release()

    print(f"[label_export] Exported {len(sorted_segments)} segments to {csv_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scored segments for labeling")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing segments_scored.json")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for CSV and keyframes")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top segments to export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = export_labels(args.run, args.out, args.top_k)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
