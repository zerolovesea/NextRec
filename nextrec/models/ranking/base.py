"""Shared base class for ranking models."""

from __future__ import annotations

from nextrec.engine.model import Model as BaseModel


class BaseRankingModel(BaseModel):
    @property
    def model_family(self) -> str:
        return "ranking"
