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

### Converting OpenAI JSONL annotations
Roboflow's "OpenAI" export stores annotations in conversational JSONL files. Convert them into the
project's processed format (normalized xyxy boxes + canonical labels) via:

```bash
source .venv/bin/activate
python scripts/convert_openai_jsonl.py            # converts both drywall + crack datasets
python scripts/convert_openai_jsonl.py --dataset cracks  # run per dataset if needed
```

Outputs land in `data/processed/<dataset>/<split>.json` with prompt text, remote `image_url`, and
label frequencies for quick sanity checks.

## Prompt Templates
Store reusable prompt sets inside `data/prompts/`. The `configs/base.yaml` file expects each entry to
have a `label` (used downstream for metrics) and a `text` field (textual description passed into the
detector). See `data/prompts/drywall_prompts.yaml` for a richer template you can copy into configs at
run time.

## Processed Dataset API
- `src/data/openai_jsonl_converter.py` parses Roboflow/OpenAI JSONL exports into normalized manifests
   (run via `scripts/convert_openai_jsonl.py`).
- `src/data/processed_dataset.py` loads those manifests and wraps each sample in a convenient
   dataclass-like record for downstream pipelines.
- `src/data/image_cache.py` plus `scripts/download_processed_images.py` let you pre-download all
   remote `image_url` assets into `data/processed_images/`:

   ```bash
   python scripts/download_processed_images.py data/processed/drywall_join_detect/train.json \
         data/processed/cracks/train.json --cache-dir data/processed_images
   ```

   Each image is stored once using a SHA-256 hash of its URL, keeping the dataset reproducible.

## Prompted Segmentation Pipeline
- `src/models/grounding_dino.py` wraps the GroundingDINO SwinT checkpoint to produce prompt-aware
   bounding boxes.
- `src/models/sam_wrapper.py` exposes a simple SAM interface to turn boxes into masks.
- `src/pipeline/prompted_segmentor.py` fuses the detector + segmentor for a single prompt.
- Run the full pipeline with:

   ```bash
   python scripts/run_prompted_segmentation.py \
         data/processed/drywall_join_detect/valid.json \
         --prompt-label exposed_joint_tape \
         --config configs/base.yaml \
         --output-dir outputs/masks/drywall_valid
   ```

   Prompt labels come from `configs/base.yaml`. Ensure the referenced GroundingDINO config, checkpoints,
   and SAM checkpoints exist under `checkpoints/` before running.

## Evaluation & Reporting
- `src/utils/mask_metrics.py` implements IoU/Dice helpers for binary masks.
- `scripts/eval_segmentation.py` compares predicted PNG masks against manifest-derived ground truth
   (boxes rasterized to masks) and reports mean IoU/Dice plus per-sample stats:

   ```bash
   python scripts/eval_segmentation.py \
         data/processed/drywall_join_detect/valid.json \
         --mask-dir outputs/smoke/drywall_joints \
         --prompt-label exposed_joint_tape \
         --target-label drywall_joint \
         --max-samples 200 \
         --image-cache data/processed_images \
         --output reports/drywall_valid_metrics.json
   ```

   Adjust `--target-label` for other datasets (e.g., `drywall_crack`) and remove `--max-samples` to
   evaluate the full split.

## Next Steps
- Implement data ingestion utilities under `src/data/`.
- Wire a prompted segmentation pipeline (GroundingDINO → SAM) under `src/pipeline/`.
- Add QA scoring/report generation logic plus regression tests.
