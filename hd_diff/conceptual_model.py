"""Conceptual HD-Diff model outline.

The classes in this file are intentionally descriptive. They help readers see
how the method is organized without exposing the full model implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .guidance import GuidanceScale, build_guidance_plan


@dataclass(frozen=True)
class HDDiffConfig:
    """High-level configuration used for method documentation."""

    modalities: Sequence[str] = ("T1", "T1ce", "T2", "FLAIR")
    target_regions: Sequence[str] = ("WT", "TC", "ET")
    feature_stages: int = 5
    denoising_steps: str = "coarse-to-fine diffusion denoising"
    notes: Sequence[str] = field(default_factory=lambda: (
        "Exact network layers are not included in the public version.",
        "Training and inference details are intentionally omitted.",
    ))


class HDDiffConcept:
    """Readable method skeleton for the public release.

    This class does not perform segmentation. It records the conceptual data
    flow used by HD-Diff so that readers can understand the framework layout.
    """

    def __init__(self, config: Optional[HDDiffConfig] = None) -> None:
        self.config = config or HDDiffConfig()
        self.guidance_plan = build_guidance_plan(self.config.feature_stages)

    def describe_inputs(self) -> Dict[str, Sequence[str]]:
        return {
            "modalities": self.config.modalities,
            "targets": self.config.target_regions,
        }

    def encode_modalities(self) -> str:
        return "extract modality-aware multi-scale features"

    def fuse_features(self) -> str:
        return "align and fuse cross-modal representations"

    def build_structure_guidance(self) -> List[GuidanceScale]:
        return self.guidance_plan

    def denoise_mask(self) -> str:
        return self.config.denoising_steps

    def pipeline(self) -> List[str]:
        """Return the public, implementation-free method flow."""

        guidance = ", ".join(
            f"stage {item.stage}:{item.role}" for item in self.guidance_plan
        )
        return [
            "input multimodal MRI volume",
            self.encode_modalities(),
            self.fuse_features(),
            f"build hierarchical guidance ({guidance})",
            self.denoise_mask(),
            "output tumor subregion mask",
        ]
