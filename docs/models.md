# Models

## Vision: DenseNet-121 (torchxrayvision)

- **Library:** `torchxrayvision==1.5.5`
- **Checkpoint:** `densenet121-res224-nih`
- **Architecture:** DenseNet-121, single-channel 224×224 input
- **Training data:** NIH ChestX-ray14
- **License (library):** Apache-2.0
- **License (weights):** `LICENSE STATUS: UNVERIFIED — the weights ship with the library, but the checkpoint's own license has not been re-verified against the torchxrayvision project.` The library as a whole is Apache-2.0; treat the weights accordingly until confirmed.
- **Permitted use for RADIANT:** Research and evaluation. Model scores are raw sigmoid outputs and are NOT calibrated probabilities.
- **Attribution:** torchxrayvision project (https://github.com/mlmed/torchxrayvision)

### Label space and mapping

torchxrayvision's `densenet121-res224-nih` checkpoint returns **18** outputs.
The 14 official NIH ChestX-ray14 labels are a subset. The 4 extra labels — `Lung Lesion`,
`Fracture`, `Lung Opacity`, `Enlarged Cardiomediastinum` — come from the
Google re-labeling and are excluded from RADIANT's `VisionResult`.

RADIANT's `backend/vision/model.py` builds an explicit index map at load time and
raises `RuntimeError` if any of the 14 NIH labels are missing from the native
pathology list. This is a hard failure, not a silent fallback.

### Calibration

`VisionResult.calibration_status` is a fixed string: `NOT CALIBRATED — raw sigmoid output`.
RADIANT does not perform temperature scaling or any other calibration. Model
scores are used as-is for prototype threshold classification only.

## LLM: AMD Token Factory (Stage 10)

Not yet implemented. Will be documented here in Stage 10 with a license entry
for the specific model used.

## Embeddings: all-MiniLM-L6-v2 (Stage 8)

Not yet implemented. Will be documented here in Stage 8.