from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from bil.io.json_utils import load_json, save_json
from bil.report import build_report


def _best_match(gold_shot: Dict, clips: List[Dict]) -> Dict:
    best = None
    best_err = float("inf")
    for clip in clips:
        start_err = abs(gold_shot["start"] - clip["start"])
        end_err = abs(gold_shot["end"] - clip["end"])
        total = start_err + end_err
        if total < best_err:
            best_err = total
            best = {
                "clip_start": clip["start"],
                "clip_end": clip["end"],
                "start_error": start_err,
                "end_error": end_err,
            }
    return best


def evaluate(gold_path: Path, run_dir: Path) -> Dict:
    gold = load_json(gold_path)
    clips_file = run_dir / "clips" / "index.json"
    clips_index = load_json(clips_file) if clips_file.exists() else []
    results: List[Dict] = []

    for video in gold.get("videos", []):
        shots = video.get("shots", [])
        for shot in shots:
            match = _best_match(shot, clips_index) if clips_index else None
            if match:
                res = {
                    "gold_start": shot["start"],
                    "gold_end": shot["end"],
                    **match,
                }
            else:
                res = {
                    "gold_start": shot["start"],
                    "gold_end": shot["end"],
                    "clip_start": None,
                    "clip_end": None,
                    "start_error": None,
                    "end_error": None,
                }
            results.append(res)

    start_errors = [r["start_error"] for r in results if r["start_error"] is not None]
    end_errors = [r["end_error"] for r in results if r["end_error"] is not None]
    stats = {
        "start_error_avg": sum(start_errors) / len(start_errors) if start_errors else None,
        "end_error_avg": sum(end_errors) / len(end_errors) if end_errors else None,
    }
    return {"results": results, "stats": stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate extracted clips against gold annotations")
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold.json")
    parser.add_argument("--runs", type=Path, required=True, help="Run directory")
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="Attach eval results to existing report.html",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_data = evaluate(args.gold, args.runs)
    save_json(Path(args.runs) / "eval.json", eval_data)
    if args.update_report:
        summary = {
            "num_videos": 0,
            "num_segments_raw": 0,
            "num_clips": 0,
            "failure_reasons": {},
        }
        build_report(Path(args.runs), summary, eval_data, logger=__import__("logging").getLogger("bil"))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
