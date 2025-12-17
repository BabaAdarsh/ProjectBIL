import subprocess
import sys
from pathlib import Path

import pytest

from bil.utils import image_io


def test_pipeline_end_to_end(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    run_dir = tmp_path / "run"
    image_io.generate_synthetic_frames(frames_dir, count=30, size=(160, 120))

    cmd = [
        sys.executable,
        "-m",
        "bil.run",
        "--input",
        str(frames_dir),
        "--out",
        str(run_dir),
        "--config",
        str(Path("configs/default.yaml").resolve()),
        "--output-format",
        "frames",
        "--input-format",
        "frames",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    expected_files = [
        run_dir / "segments_raw.json",
        run_dir / "segments_scored.json",
        run_dir / "clips" / "index.json",
        run_dir / "report.html",
        run_dir / "run.log",
    ]
    for file in expected_files:
        assert file.exists(), f"Missing output file: {file}"


def test_label_export_frames(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    run_dir = tmp_path / "run"
    export_dir = tmp_path / "export"
    image_io.generate_synthetic_frames(frames_dir, count=40, size=(120, 90))

    run_cmd = [
        sys.executable,
        "-m",
        "bil.run",
        "--input",
        str(frames_dir),
        "--out",
        str(run_dir),
        "--config",
        str(Path("configs/default.yaml").resolve()),
        "--output-format",
        "frames",
        "--input-format",
        "frames",
    ]
    run_result = subprocess.run(run_cmd, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr

    export_cmd = [
        sys.executable,
        "-m",
        "bil.label_export",
        "--run",
        str(run_dir),
        "--out",
        str(export_dir),
        "--top-k",
        "5",
    ]
    export_result = subprocess.run(export_cmd, capture_output=True, text=True)
    assert export_result.returncode == 0, export_result.stderr

    csv_path = export_dir / "segments.csv"
    keyframes_dir = export_dir / "keyframes"
    assert csv_path.exists(), "segments.csv missing after label export"
    montage_files = list(keyframes_dir.glob("segment_*.png"))
    assert montage_files, "No keyframe montages were created"


def test_pipeline_mp4_if_cv2(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_path = tmp_path / "input.mp4"
    run_dir = tmp_path / "run"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for i in range(45):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = (i * 5) % 255
        writer.write(frame)
    writer.release()

    cmd = [
        sys.executable,
        "-m",
        "bil.run",
        "--input",
        str(video_path),
        "--out",
        str(run_dir),
        "--config",
        str(Path("configs/default.yaml").resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (run_dir / "clips" / "index.json").exists()
