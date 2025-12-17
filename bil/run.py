from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from bil.io.json_utils import save_json
from bil.io.schema import FailureCode, Segment, ScoredSegment
from bil.pipeline import ingest, shot_boundary, batting_camera, tracker, clip_writer
from bil.report import build_report


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("bil")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        import ast

        cfg: Dict[str, object] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    cfg[key] = ast.literal_eval(value)
                except Exception:
                    cfg[key] = value
        return cfg
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def orchestrate(input_path: Path, run_dir: Path, config_path: Path, output_format: str) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    logger.info("Starting pipeline. input=%s, run_dir=%s", input_path, run_dir)

    config = load_config(config_path)
    save_json(run_dir / "config_resolved.json", config)

    sources = ingest.collect_sources(input_path)
    if not sources:
        logger.error("No input sources found at %s", input_path)
        return 1

    segments_all: List[Segment] = []
    scored_all: List[ScoredSegment] = []
    tracks_all = []
    clip_index: List[Dict] = []
    failure_reasons: Dict[str, int] = {}

    for source in sources:
        logger.info("Processing input: %s (%s)", source.path, source.kind)
        segments, decode_ok = shot_boundary.detect_shots(source, config, logger)
        if not decode_ok:
            failure_reasons[FailureCode.VIDEO_DECODE_FAIL.value] = (
                failure_reasons.get(FailureCode.VIDEO_DECODE_FAIL.value, 0) + 1
            )
        if not segments:
            failure_reasons[FailureCode.NO_SEGMENTS.value] = (
                failure_reasons.get(FailureCode.NO_SEGMENTS.value, 0) + 1
            )
            continue
        segments_all.extend(segments)

        scored = batting_camera.score_segments(source, segments, config, logger)
        scored_all.extend(scored)
        threshold = float(config.get("score_threshold", 0.0))
        selected = [s for s in scored if s.score >= threshold]
        if not selected:
            failure_reasons[FailureCode.LOW_SCORE.value] = (
                failure_reasons.get(FailureCode.LOW_SCORE.value, 0) + 1
            )
            continue

        tracks = tracker.track_segments(source, selected, config, logger)
        if not tracks:
            failure_reasons[FailureCode.TRACK_FAIL.value] = (
                failure_reasons.get(FailureCode.TRACK_FAIL.value, 0) + 1
            )
            continue
        tracks_all.extend(tracks)

        clips_meta = clip_writer.write_clips(source, tracks, config, run_dir, logger, output_format=output_format)
        if not clips_meta:
            failure_reasons[FailureCode.CLIP_WRITE_FAIL.value] = (
                failure_reasons.get(FailureCode.CLIP_WRITE_FAIL.value, 0) + 1
            )
        for meta in clips_meta:
            if meta.get("failure"):
                failure_reasons[meta["failure"]] = failure_reasons.get(meta["failure"], 0) + 1
            else:
                clip_index.append(meta)

    save_json(run_dir / "segments_raw.json", [s.to_dict() for s in segments_all])
    save_json(
        run_dir / "segments_scored.json",
        [
            {
                "video": s.video,
                "start": s.start,
                "end": s.end,
                "score": s.score,
                "reason": s.reason,
            }
            for s in scored_all
        ],
    )
    from bil.io.schema import serialize_tracks

    save_json(run_dir / "tracks" / "tracks.json", serialize_tracks(tracks_all))
    save_json(run_dir / "clips" / "index.json", clip_index)

    summary = {
        "num_videos": len(sources),
        "num_segments_raw": len(segments_all),
        "num_clips": len(clip_index),
        "failure_reasons": failure_reasons,
    }
    build_report(run_dir, summary, eval_data=None, logger=logger)
    logger.info("Pipeline completed. Summary: %s", summary)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProjectBIL extractor skeleton")
    parser.add_argument("--input", type=Path, required=True, help="Path to input video or folder")
    parser.add_argument("--out", type=Path, required=True, help="Run directory for outputs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML configuration file",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "mp4", "frames"],
        default="auto",
        help="Output clip format. Auto picks mp4 when OpenCV is available, otherwise frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = orchestrate(args.input, args.out, args.config, output_format=args.output_format)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
