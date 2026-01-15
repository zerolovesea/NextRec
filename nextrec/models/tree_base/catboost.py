"""
CatBoost adapter for NextRec.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nextrec.models.tree_base.base import TreeBaseModel


class Catboost(TreeBaseModel):
    model_file_suffix = "cbm"

    @property
    def model_name(self) -> str:
        return "catboost"

    def build_estimator(self, model_params: dict[str, Any], epochs: int | None):
        try:
            import catboost as cb
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "catboost is required for model='catboost'. Install with: pip install catboost"
            ) from exc
        params = dict(model_params)
        if epochs is not None and "iterations" not in params:
            params["iterations"] = epochs
        task = self.task[0] if isinstance(self.task, list) else self.task
        if task == "regression":
            params.setdefault("loss_function", "RMSE")
            return cb.CatBoostRegressor(**params)
        if task == "binary":
            params.setdefault("loss_function", "Logloss")
            return cb.CatBoostClassifier(**params)
        raise ValueError(f"[catboost-init Error] Unsupported task type: {task}")

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
        X_train, cat_features = self._prepare_catboost_data(X_train, cat_features)
        if cat_features:
            fit_kwargs.setdefault("cat_features", cat_features)
        if X_valid is not None and y_valid is not None:
            X_valid, _ = self._prepare_catboost_data(X_valid, cat_features)
            fit_kwargs.setdefault("eval_set", (X_valid, y_valid))
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def save_model_file(self, model: Any, path) -> None:
        model.save_model(str(path))

    def load_model_file(self, path):
        model = self.build_estimator(dict(self.model_params), epochs=None)
        model.load_model(str(path))
        return model

    def _prepare_catboost_data(
        self, X: Any, cat_features: list[int]
    ) -> tuple[Any, list[int]]:
        if not cat_features:
            return X, cat_features
        if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating):
            feature_names = [f"f{idx}" for idx in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            for idx in cat_features:
                if 0 <= idx < df.shape[1]:
                    series = df.iloc[:, idx]
                    df.iloc[:, idx] = series.map(
                        lambda value: (
                            ""
                            if pd.isna(value)
                            else (
                                str(int(value))
                                if isinstance(value, (float, np.floating))
                                and value.is_integer()
                                else str(value)
                            )
                        )
                    )
            return df, cat_features
        return X, cat_features

    def predict_scores(self, model: Any, X: np.ndarray) -> np.ndarray:
        X, _ = self._prepare_catboost_data(X, self._cat_feature_indices)
        return super().predict_scores(model, X)
