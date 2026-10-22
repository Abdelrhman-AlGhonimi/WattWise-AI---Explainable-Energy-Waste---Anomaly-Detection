from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import logging


def model_factory(seed=42):
    models = {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
        # Leaf/depth constraints reduce memorization of isolated high-use spikes.
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=14,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=6,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        ),
        # Absolute loss targets the primary MAE objective and is less sensitive to rare spikes.
        "HistGradientBoosting": HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=5,
            early_stopping=False,
            random_state=seed,
        ),
        "HistGradientBoosting (squared loss)": HistGradientBoostingRegressor(
            max_iter=106,
            learning_rate=0.0346,
            max_leaf_nodes=17,
            l2_regularization=0.39,
            early_stopping=False,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBRegressor

        # Pseudo-Huber's slope is scaled to the dataset's Wh range; its default
        # slope of 1 saturates gradients on this target and severely underfits.
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5,
            objective="reg:pseudohubererror",
            huber_slope=50,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    except (ImportError, OSError) as exc:
        logging.warning("Optional XGBoost model unavailable: %s", exc)
    return models
