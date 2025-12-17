from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from bil.io.json_utils import save_json, load_json


def build_report(run_dir: Path, summary: Dict, eval_data: Optional[Dict], logger: logging.Logger) -> None:
    report_path = run_dir / "report.html"
    eval_path = run_dir / "eval.json"
    if eval_data:
        save_json(eval_path, eval_data)

    num_videos = summary.get("num_videos", 0)
    num_segments = summary.get("num_segments_raw", 0)
    num_clips = summary.get("num_clips", 0)
    failure_reasons = summary.get("failure_reasons", {})
    frames_fps = summary.get("frames_fps")
    output_fps = summary.get("output_fps")
    input_kind = summary.get("input_kind")
    avg_clip_dur = 0.0
    clips_index = run_dir / "clips" / "index.json"
    if clips_index.exists():
        clips = load_json(clips_index)
        if clips:
            durations = [c["end"] - c["start"] for c in clips]
            avg_clip_dur = sum(durations) / len(durations)

    eval_section = ""
    if eval_path.exists():
        eval_json = load_json(eval_path)
        stats = eval_json.get("stats", {})
        eval_section = f"""
        <h2>Evaluation</h2>
        <pre>{eval_json}</pre>
        <p>Average start error: {stats.get('start_error_avg', 'n/a')}</p>
        <p>Average end error: {stats.get('end_error_avg', 'n/a')}</p>
        """

    failure_list = "".join(
        f"<li>{code}: {count}</li>" for code, count in sorted(failure_reasons.items(), key=lambda x: x[0])
    )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ProjectBIL Report</title></head>
<body>
<h1>ProjectBIL Run Report</h1>
<ul>
  <li>Input videos: {num_videos}</li>
  <li>Raw segments: {num_segments}</li>
  <li>Final clips: {num_clips}</li>
  <li>Average clip duration (s): {avg_clip_dur:.2f}</li>
  <li>Input type: {input_kind or 'unknown'}</li>
  <li>Frames FPS (frames mode): {frames_fps if frames_fps is not None else 'n/a'}</li>
  <li>Output FPS: {output_fps if output_fps is not None else 'n/a'}</li>
</ul>
<h2>Failure reasons</h2>
<ul>{failure_list or '<li>None</li>'}</ul>
{eval_section}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", report_path)
