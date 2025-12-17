## ProjectBIL

Extractor MVP skeleton for cricket batting segments.

### Setup (Windows-friendly)
- Requires Python 3.9+ (CPU-only; GPU not required).
- Create and activate a virtual environment:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- Install dependencies without build isolation:
  ```powershell
  pip install -r requirements-dev.txt
  # Optional for mp4/video support:
  pip install opencv-python numpy pillow
  ```

### Run the pipeline
```powershell
python -m bil.run --input <path_to_video_or_folder_or_frames_dir> --out run_dir --config configs/default.yaml --input-format auto --output-format auto
```

Outputs (segments, tracks, clips, report) will appear under `run_dir`.
Use `--input-format frames --output-format frames` to run without OpenCV using a directory of PNG/JPG frames. For MP4 in/out, install OpenCV and keep `auto` or set `--input-format video --output-format mp4`.

### Evaluation
```powershell
python -m bil.eval --gold path/to/gold.json --runs run_dir --update-report
```

### Tests (fast, synthetic video)
```powershell
pytest
```

Notes: CPU-only MVP skeleton; heuristics are lightweight and deterministic on the synthetic test video. Future improvements can swap in stronger detectors and pose models without changing the interfaces.
