# Prompt-Based Drywall QA Segmentation

Production-ready pipeline for drywall quality assurance. Each defect type (joints vs cracks) gets its
own fine-tuned DeepLabV3-ResNet50 head. Manifests describe the imagery, the router picks the correct
checkpoint, and automated evaluation reports IoU/Dice so QA teams can catch exposed tape, skim-coat
issues, and cracks early.

## Highlights
- DeepLab-only stack: lightweight checkpoints per defect with a single, consistent inference path.
- Flexible routing: `configs/segmentation_routes.yaml` maps prompt labels to checkpoints and output
  suffix conventions so all CLIs stay consistent.
- Single-image + batch inference: iterate a manifest with the router or run targeted experiments with
  `scripts/run_single_prompt_inference.py`.
- Reproducible evaluation: prompts are sanitized into deterministic suffixes so metrics can always be
  tied back to the masks that generated them.

**Full report**: see `full_pipeline_report.pdf` for the choices made summary, Visual Examples, data splits, runtime/footprint stats, metrics, and failure notes.

## Repository Layout
```
├── checkpoints/                 # fine-tuned DeepLab weights (.pth)
├── configs/
│   └── segmentation_routes.yaml # prompt-label routing map
├── data/
│   ├── processed/               # manifests (train/valid/test JSON)
│   ├── processed_images/        # cached RGB assets keyed by SHA-256
│   └── raw/                     # Roboflow exports + README.roboflow.txt
├── outputs/
│   ├── deeplab*/                # batched DeepLab runs
│   ├── manual/                  # ad-hoc single-image masks
│   └── routed/                  # router-driven experiments
├── reports/                     # metrics JSON (IoU/Dice histories & evals)
├── scripts/                     # CLI entry points (train, router, eval, etc.)
├── src/
│   ├── data/                    # manifests, cache, box-mask datasets
│   ├── pipeline/                # DeepLab inference helpers
│   └── utils/                   # evaluation + metrics utilities
└── tests/
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Populate `data/raw/` with Roboflow exports, keep checkpoints in `checkpoints/`, and download/copy any
pretrained weights referenced by the router.

## Data Preparation
1. **Download Roboflow exports**
   ```bash
   curl -L "https://app.roboflow.com/ds/<dataset>?key=<api-key>" -o data/raw/drywall.zip
   cd data/raw && unzip -o drywall.zip && rm drywall.zip
   ```
2. **Convert COCO → processed manifests**
   ```bash
   source .venv/bin/activate
   python scripts/convert_coco.py --dataset all
   ```
   This walks each Roboflow `coco_*` export (train/valid/test), normalizes boxes, and writes manifests to `data/processed/<dataset>/<split>.json`.
3. **Pre-cache imagery for offline use**
   ```bash
   python scripts/download_processed_images.py \
      data/processed/drywall_join_detect/train.json \
      data/processed/cracks/train.json \
      --cache-dir data/processed_images
   ```

## Pipeline Overview
```
Roboflow export → convert_coco.py → ProcessedDataset JSON
        ↓                                      ↓
download_processed_images.py          box_mask_dataset.py
        ↓                                      ↓
cached RGBs (data/processed_images)    train_segmentation.py → checkpoints/*.pth
        ↓                                      ↓
routes YAML ──┐                     deeplab_inference.py helpers
              ├─ run_segmentation_router.py ─┐
              ├─ run_deeplab_segmentation.py ├─ PNG masks (outputs/*)
              └─ run_single_prompt_inference.py
                                                ↓
                                     eval_segmentation.py → reports/*.json
```

### Inference pipeline internals
At inference time the router, single-image CLI, and evaluator all follow the same text-handling and
checkpoint-selection logic:

1. **CLI input** – you pass `--prompt-label <logical-name>` plus a free-form `--prompt-text`.
2. **Route lookup** – `configs/segmentation_routes.yaml` maps the prompt label to a checkpoint,
   resize/batch overrides, class labels, and an optional mask suffix.
3. **Sanitize prompt text** – `sanitize_prompt_text` lowercases, replaces spaces with `_`, and removes
   punctuation so filenames stay deterministic (e.g., "Highlight exposed drywall joints" →
   `highlight_exposed_drywall_joints`). This sanitized text is concatenated with the label name unless
   `--no-label-suffix` is provided.
4. **DeepLab configuration** – `src/pipeline/deeplab_inference.py` loads the checkpoint, resolves the
   active class labels, and builds label → channel index mappings to extract the requested masks.
5. **Mask naming** – each PNG follows `<sample_id>__<sanitized_prompt>_<label>.png`, ensuring
   evaluation CLIs can rediscover the files later just by repeating the same prompt text.
6. **Evaluation** – `scripts/eval_segmentation.py` sanitizes the provided `--prompt-text` the exact
   same way before searching `--mask-dir`, so as long as you copy the text emitted by the inference
   CLI, the evaluator finds the masks automatically.

```
[CLI args] → [routes YAML lookup] → [DeepLab config + checkpoint]
       ↓                      ↓
 sanitize(prompt_text)   resolve class labels
       ↓                      ↓
 build mask filename suffix  model forward pass
                 \            /
                  → write PNG masks → eval_segmentation.py (same sanitize())
```

Use the JSON summary printed by `run_segmentation_router.py` or `run_single_prompt_inference.py` to
copy the exact `prompt_suffixes` value into downstream evaluation commands.

## Core CLIs

### Training
```bash
python scripts/train_segmentation.py \
   data/processed/drywall_join_detect/train.json \
   data/processed/drywall_join_detect/valid.json \
   --target-label drywall_joint --epochs 6 --batch-size 4 \
   --resize 512 512 --cached-only --device cuda \
   --output checkpoints/deeplabv3_joint_latest.pth \
   --metrics-output reports/finetune_deeplab_joint_history.json
```
- Uses `src/data/box_mask_dataset.py` to rasterize boxes into binary masks on the fly.
- Metrics JSON stores IoU/Dice per epoch so you can monitor regressions.

### Batch inference via router
```bash
python scripts/run_segmentation_router.py \
   data/processed/drywall_join_detect/valid.json \
   --prompt-label joint_latest \
   --output-dir outputs/routed/joint_latest_long_prompt \
   --max-samples 100 --skip-existing
```
- Router looks up `joint_latest` inside `configs/segmentation_routes.yaml` to find the checkpoint,
  resize, batch size, and mask suffix.
- Add new defect types by extending the YAML with `{checkpoint, mask_suffix, class_labels}`.

### Single-image inference
```bash
python scripts/run_single_prompt_inference.py \
   data/processed_images/<sha>.jpg \
   --prompt-label crack_latest \
   --prompt-text "Outline fine surface cracks" \
   --output-dir outputs/manual/crack_test \
   --output-name crack_demo --no-label-suffix
```
- Sanitized prompt text defines the filename suffix that `eval_segmentation.py` expects.
- `--no-label-suffix` keeps the suffix compact when you only emit one class.

### Evaluation
```bash
python scripts/eval_segmentation.py \
   data/processed/drywall_join_detect/valid.json \
   --mask-dir outputs/routed/joint_latest_long_prompt \
   --prompt-label joint_latest \
   --prompt-text joint_latest_drywall_joint \
   --target-label drywall_joint \
   --image-cache data/processed_images \
   --max-samples 5 \
   --output reports/joint_latest_long_prompt_eval.json
```
- `--prompt-text` **must** match the sanitized suffix used by inference (see CLI output) so metrics
  can locate the correct PNGs.
- Outputs include per-sample IoU/Dice plus aggregated means.

## Results (latest runs)
| Defect Type   | Dataset Split | Prompt Label | Mean IoU | Mean Dice | Samples | Report |
|--------------|---------------|--------------|---------:|----------:|--------:|--------|
| Drywall Joint | drywall_join_detect / valid | `joint_latest` | **0.803** | **0.880** | 202 | `reports/joint_latest_eval.json` |
| Drywall Crack | cracks / valid | `crack_latest` | **0.667** | **0.768** | 201 | `reports/crack_latest_eval.json` |

Full validation splits were used for these numbers (router-generated masks evaluated via `scripts/eval_segmentation.py`).

## Visula Examples

<table class="visual-grid">
   <thead>
      <tr>
         <th>Sample</th>
         <th>Original</th>
         <th>Ground Truth</th>
         <th>Prediction</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><code>cracks_valid_00121</code></td>
         <td><img src="reports/examples/cracks_valid_00121_original.png" width="200" /></td>
         <td><img src="reports/examples/cracks_valid_00121_gt.png" width="200" /></td>
         <td><img src="reports/reports/examples/cracks_valid_00121_prediction.png" width="200" /></td>
      </tr>
      <tr>
         <td><code>cracks_valid_00118</code></td>
         <td><img src="reports/examples/cracks_valid_00118_original.png" width="200" /></td>
         <td><img src="reports/examples/cracks_valid_00118_gt.png" width="200" /></td>
         <td><img src="reports/examples/cracks_valid_00118_prediction.png" width="200" /></td>
      </tr>
      <tr>
         <td><code>cracks_valid_00096</code></td>
         <td><img src="reports/examples/cracks_valid_00096_original.png" width="200" /></td>
         <td><img src="reports/examples/cracks_valid_00096_gt.png" width="200" /></td>
         <td><img src="reports/examples/cracks_valid_00096_prediction.png" width="200" /></td>
      </tr>
      <tr>
         <td><code>drywall_join_detect_valid_00093</code></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00093_original.png" width="200" /></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00093_gt.png" width="200" /></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00093_prediction.png" width="200" /></td>
      </tr>
      <tr>
         <td><code>drywall_join_detect_valid_00141</code></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00141_original.png" width="200" /></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00141_gt.png" width="200" /></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00141_prediction.png" width="200" /></td>
      </tr>
      <tr>
         <td><code>drywall_join_detect_valid_00123</code></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00123_original.png" width="200" /></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00123_gt.png" width="200" /></td>
         <td><img src="reports/examples/drywall_join_detect_valid_00123_prediction.png" width="200" /></td>
      </tr>
   </tbody>
</table>

## Implementation Notes
- `src/pipeline/deeplab_inference.py` now exposes both batched-manifest and single-image helpers so
  every CLI shares the same preprocessing + checkpoint loading code path.
- Prompt text is sanitized via `src/utils/evaluator.sanitize_prompt_text`, so long natural-language
  descriptions can be used consistently across inference and evaluation.

## Next Steps
1. Scale evaluation to the full validation/test manifests for statistically stable metrics.
2. Extend the router with additional defect prompts (e.g., nail pops, paint runs) once checkpoints
   are available.
3. Add automated regression tests under `tests/` for the new single-image CLI and evaluator sanity
   checks.
