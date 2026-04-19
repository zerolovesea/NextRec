"""Shared base class for multitask models."""

from __future__ import annotations

from nextrec.engine.model import Model as BaseModel


class BaseMultitaskModel(BaseModel):
    @property
    def model_family(self) -> str:
        return "multitask"
