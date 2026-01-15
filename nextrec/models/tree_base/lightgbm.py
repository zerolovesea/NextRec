"""
LightGBM adapter for NextRec.
"""

from __future__ import annotations

from typing import Any

from nextrec.models.tree_base.base import TreeBaseModel


class Lightgbm(TreeBaseModel):
    model_file_suffix = "txt"

    @property
    def model_name(self) -> str:
        return "lightgbm"

    def build_estimator(self, model_params: dict[str, Any], epochs: int | None):
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "lightgbm is required for model='lightgbm'. Install with: pip install lightgbm"
            ) from exc
        params = dict(model_params)
        if epochs is not None and "n_estimators" not in params:
            params["n_estimators"] = epochs
        task = self.task[0] if isinstance(self.task, list) else self.task
        if task == "regression":
            params.setdefault("objective", "regression")
            return lgb.LGBMRegressor(**params)
        if task == "binary":
            params.setdefault("objective", "binary")
            return lgb.LGBMClassifier(**params)
        raise ValueError(f"[lightgbm-init Error] Unsupported task type: {task}")

    def fit_estimator(
        self,
        model: Any,
        X_train,
        y_train,
        X_valid,
        y_valid,
        cat_features,
        **kwargs,
    ):
        fit_kwargs = dict(kwargs)
        if cat_features:
            fit_kwargs.setdefault("categorical_feature", cat_features)
        if X_valid is not None and y_valid is not None:
            fit_kwargs.setdefault("eval_set", [(X_valid, y_valid)])
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def save_model_file(self, model: Any, path) -> None:
        if hasattr(model, "booster_"):
            model.booster_.save_model(str(path))
        else:
            model.save_model(str(path))

    def load_model_file(self, path):
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "lightgbm is required for model='lightgbm'. Install with: pip install lightgbm"
            ) from exc
        return lgb.Booster(model_file=str(path))
