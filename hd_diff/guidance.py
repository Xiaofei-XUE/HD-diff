"""Structure-guidance plan used in the public HD-Diff overview."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GuidanceScale:
    """A conceptual guidance assignment for one feature stage."""

    stage: int
    role: str
    purpose: str


def build_guidance_plan(feature_stages: int = 5) -> List[GuidanceScale]:
    """Create a high-level boundary/core guidance plan.

    The returned objects document the design intent only. They are not attention
    modules and do not contain segmentation logic.
    """

    plan: List[GuidanceScale] = []
    for stage in range(feature_stages):
        if stage < 2:
            plan.append(GuidanceScale(
                stage=stage,
                role="boundary",
                purpose="preserve local contour and edge-sensitive details",
            ))
        else:
            plan.append(GuidanceScale(
                stage=stage,
                role="core",
                purpose="support semantic consistency of tumor-core regions",
            ))
    return plan
