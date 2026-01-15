"""
XGBoost adapter for NextRec.
"""

from __future__ import annotations

from typing import Any

from nextrec.models.tree_base.base import TreeBaseModel


class Xgboost(TreeBaseModel):
    model_file_suffix = "json"

    @property
    def model_name(self) -> str:
        return "xgboost"

    def build_estimator(self, model_params: dict[str, Any], epochs: int | None):
        try:
            import xgboost as xgb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "xgboost is required for model='xgboost'. Install with: pip install xgboost"
            ) from exc
        params = dict(model_params)
        if epochs is not None and "n_estimators" not in params:
            params["n_estimators"] = epochs
        task = self.task[0] if isinstance(self.task, list) else self.task
        if task == "regression":
            params.setdefault("objective", "reg:squarederror")
            return xgb.XGBRegressor(**params)
        if task == "binary":
            params.setdefault("objective", "binary:logistic")
            return xgb.XGBClassifier(**params)
        raise ValueError(f"[xgboost-init Error] Unsupported task type: {task}")

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
        del cat_features  # xgboost handles categorical features via params
        fit_kwargs = dict(kwargs)
        if X_valid is not None and y_valid is not None:
            fit_kwargs.setdefault("eval_set", [(X_valid, y_valid)])
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def save_model_file(self, model: Any, path) -> None:
        model.save_model(str(path))

    def load_model_file(self, path):
        model = self.build_estimator(dict(self.model_params), epochs=None)
        model.load_model(str(path))
        return model
