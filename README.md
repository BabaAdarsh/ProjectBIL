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
python -m bil.run --input <path_to_video_or_folder> --out run_dir --config configs/default.yaml --output-format auto
```

Outputs (segments, tracks, clips, report) will appear under `run_dir`.

### Evaluation
```powershell
python -m bil.eval --gold path/to/gold.json --runs run_dir --update-report
```

### Tests (fast, synthetic video)
```powershell
pytest
```

Notes: CPU-only MVP skeleton; heuristics are lightweight and deterministic on the synthetic test video. Future improvements can swap in stronger detectors and pose models without changing the interfaces.
