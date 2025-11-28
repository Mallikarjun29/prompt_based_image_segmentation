# Prompt-Based Drywall QA Segmentation

Prompt-driven segmentation pipeline for drywall quality assurance images. The project will combine an
open-vocabulary detector (e.g., GroundingDINO) with a high-resolution segmentor (e.g., SAM/HQ-SAM) to
mask drywall defects such as exposed tape, screw pops, and joint cracks, then compute QA metrics for
flagging issues.

## Repository Layout
- `configs/` – experiment + evaluation settings.
- `data/` – `raw/`, `processed/`, and `prompts/` folders for assets (kept empty with gitkeep files).
- `notebooks/` – exploratory analysis, visualization, and reporting notebooks.
- `reports/` – generated QA summaries, charts, or exportable artefacts.
- `scripts/` – CLI utilities for training/inference automation.
- `src/` – reusable Python packages (`data/`, `models/`, `pipeline/`, `utils/`).
- `tests/` – unit and integration tests guarding data transforms and QA metrics.

## Getting Started
1. Create the virtual environment if it does not exist yet:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Place drywall imagery inside `data/raw/` and update prompt templates in `data/prompts/`.
4. Copy the base configuration from `configs/` and update dataset paths + prompt definitions.

## Roboflow Datasets
You already host drywall QA imagery (standard drywall + crack-focused sets) in Roboflow. Export each
project as "Segmentation" format with PNG/JPEG images plus COCO annotations, then download straight
into `data/raw/`:

```bash
# Example shared export link (replace with your Roboflow download URL)
curl -L "https://app.roboflow.com/ds/<dataset-id>?key=<api-key>" -o data/raw/drywall.zip
cd data/raw && unzip -o drywall.zip && rm drywall.zip
```

Keep crack-only exports in their own folder (e.g., `data/raw/cracks/`) so augmentation scripts can
balance classes. Update `configs/base.yaml` → `paths.raw_data_dir` if you prefer a different layout.

## Prompt Templates
Store reusable prompt sets inside `data/prompts/`. The `configs/base.yaml` file expects each entry to
have a `label` (used downstream for metrics) and a `text` field (textual description passed into the
detector). See `data/prompts/drywall_prompts.yaml` for a richer template you can copy into configs at
run time.

## Next Steps
- Implement data ingestion utilities under `src/data/`.
- Wire a prompted segmentation pipeline (GroundingDINO → SAM) under `src/pipeline/`.
- Add QA scoring/report generation logic plus regression tests.
