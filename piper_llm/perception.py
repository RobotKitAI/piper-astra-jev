"""Local open-vocabulary detection: Grounding DINO or SAM 3.

Both take text prompts and return boxes. Grounding DINO is faster; SAM 3 finds
more of the small features, such as bolt holes.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

DINO_REPO = "IDEA-Research/grounding-dino-tiny"
SAM3_REPO = "facebook/sam3"  # gated model: needs HF_TOKEN


@dataclass
class Detection:
    """One detection: image centre, box and score."""

    phrase: str
    centre: tuple[float, float]
    box: tuple[float, float, float, float]
    score: float
    mask: Any = None  # SAM 3 only: uint8 image mask of the instance


@dataclass
class Detector:
    """Text-prompted detector on the local GPU."""

    backend: str = "dino"  # "dino" or "sam3"
    threshold: float = 0.25
    latencies: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        import torch

        # A busy host can spend hundreds of ms in preprocessing while the GPU
        # idles: cap the thread pool and use the fast image processor.
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self._torch = torch
        if self.backend == "dino":
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.proc = AutoProcessor.from_pretrained(DINO_REPO, use_fast=True)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_REPO, dtype=torch.float16).to("cuda").eval()
        else:
            from transformers import Sam3Model, Sam3Processor

            self.proc = Sam3Processor.from_pretrained(SAM3_REPO)
            self.model = Sam3Model.from_pretrained(
                SAM3_REPO, dtype=torch.float16).to("cuda").eval()

    def detect(self, rgb: npt.NDArray[np.uint8], phrases: list[str]) -> list[Detection]:
        """All detections for the given phrases, best score first."""
        img = Image.fromarray(rgb)
        t0 = time.time()
        out = (self._detect_dino(img, phrases) if self.backend == "dino"
               else self._detect_sam3(img, phrases))
        self.latencies.append(time.time() - t0)
        return sorted(out, key=lambda d: -d.score)

    def best(self, rgb: npt.NDArray[np.uint8], phrase: str,
             max_px: float | None = None) -> Detection | None:
        """Highest-scoring detection of one phrase, or None.

        max_px drops boxes wider or taller than that: when the gripper hides
        the cube, the best "red cube" is otherwise the tray.
        """
        hits = [d for d in self.detect(rgb, [phrase]) if d.phrase == phrase
                and (max_px is None or max(d.box[2] - d.box[0], d.box[3] - d.box[1]) <= max_px)]
        return hits[0] if hits else None

    def _detect_dino(self, img: Image.Image, phrases: list[str]) -> list[Detection]:
        torch = self._torch
        text = ". ".join(p.lower() for p in phrases) + "."
        inputs = self.proc(images=img, text=text, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            raw = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            raw, inputs.input_ids.cpu(), threshold=self.threshold, text_threshold=0.2,
            target_sizes=[img.size[::-1]])[0]
        out = []
        for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
            b = box.tolist()
            match = next((p for p in phrases if p.split()[-1] in str(label).lower()), None)
            if match:
                out.append(Detection(match, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                                     (b[0], b[1], b[2], b[3]), float(score)))
        return out

    def _detect_sam3(self, img: Image.Image, phrases: list[str]) -> list[Detection]:
        torch = self._torch
        out = []
        for phrase in phrases:  # SAM 3 takes one concept per pass
            inputs = self.proc(images=img, text=phrase, return_tensors="pt").to("cuda")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.inference_mode():
                raw = self.model(**inputs)
            res = self.proc.post_process_instance_segmentation(
                raw, threshold=0.4, target_sizes=[img.size[::-1]])[0]
            masks = res.get("masks")
            for i, (box, score) in enumerate(zip(res["boxes"].float().tolist(),
                                                 res["scores"].float().tolist())):
                mask = None if masks is None else masks[i].cpu().numpy().astype(np.uint8)
                out.append(Detection(phrase, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                                     tuple(box), float(score), mask))
        return out

    def median_ms(self) -> float:
        """Median detection latency so far."""
        return float(np.median(self.latencies) * 1000) if self.latencies else float("nan")


def cluster(points: list[Any], min_gap_m: float = 0.015) -> list[Any]:
    """Drop detections that land on top of each other in 3D."""
    kept: list[Any] = []
    for p in points:
        if all(float(np.linalg.norm(np.asarray(p) - np.asarray(k))) > min_gap_m for k in kept):
            kept.append(p)
    return kept
