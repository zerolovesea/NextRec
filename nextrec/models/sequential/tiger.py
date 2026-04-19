"""
Tiger sequential generative matching model.

The implementation lives under the sequential family because Tiger models
user behavior sequences and predicts the next target item/token with a
generative objective.
"""

from __future__ import annotations

from nextrec.models.sequential.sasrec import SASRec


class Tiger(SASRec):
    """
    Temporary Tiger implementation built on top of the SASRec backbone.

    This keeps the public model registry stable while the full Tiger-specific
    architecture is not yet integrated into the NextRec training stack.
    """

    @property
    def model_name(self) -> str:
        return "Tiger"
