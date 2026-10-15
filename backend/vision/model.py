"""
Vision model: torchxrayvision DenseNet-121 (densenet121-res224-nih).

Model scores are raw sigmoid outputs, NOT calibrated probabilities
(rule 3.14). The output labels are mapped from torchxrayvision's native
18-label space to the 14 official NIH ChestX-ray14 labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
import torchxrayvision as xrv

from backend.core.config import VISION_CHECKPOINT
from backend.core.logging import get_logger
from backend.vision.preprocess import preprocess
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult
from backend.vision.thresholds import classify

logger = get_logger(__name__)

# torchxrayvision's DenseNet-121 (trained on NIH) returns 18 outputs.
# These are the 4 extra labels that are NOT part of the 14 official NIH labels.
# We exclude them explicitly when mapping to NIH_LABELS.
_XRV_EXTRA_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Lung Lesion",
        "Fracture",
        "Lung Opacity",
        "Enlarged Cardiomediastinum",
    }
)


class VisionModel:
    """Wrapper around torchxrayvision DenseNet-121."""

    def __init__(self, checkpoint: str = VISION_CHECKPOINT) -> None:
        self.checkpoint = checkpoint
        logger.info("loading vision checkpoint: %s", checkpoint)
        self._model = xrv.models.DenseNet(weights=checkpoint)
        self._model.eval()
        # Native label order from torchxrayvision's pathology list.
        self._xrv_pathologies: list[str] = list(self._model.pathologies)
        # Build the mapping from NIH label -> index in the xrv output vector.
        self._nih_index: dict[str, int] = {}
        for label in NIH_LABELS:
            if label not in self._xrv_pathologies:
                raise RuntimeError(
                    f"NIH label '{label}' not found in torchxrayvision pathologies: "
                    f"{self._xrv_pathologies}"
                )
            self._nih_index[label] = self._xrv_pathologies.index(label)
        logger.info("vision model ready; mapped %d NIH labels", len(self._nih_index))

    @torch.no_grad()
    def predict(self, image_path: Path | str) -> VisionResult:
        """Run inference on a single image. Returns a VisionResult."""
        tensor = preprocess(image_path)
        outputs = self._model(tensor)
        scores = outputs[0].detach().cpu().numpy()

        findings: list[VisionFinding] = []
        for label in NIH_LABELS:
            score = float(scores[self._nih_index[label]])
            findings.append(
                VisionFinding(
                    name=label,
                    model_score=score,
                    status=classify(score),
                )
            )

        return VisionResult(
            findings=findings,
            checkpoint=self.checkpoint,
            notes=[
                "Model scores are raw sigmoid outputs. NOT calibrated probabilities.",
                "Thresholds are UNVALIDATED PROTOTYPE HEURISTICS.",
                "Extra torchxrayvision labels excluded: "
                + ", ".join(sorted(_XRV_EXTRA_LABELS)),
            ],
        )