from __future__ import annotations
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from .evaluation import regression_metrics


def optimize(X, y, trials=20, seed=42):
    try:
        import optuna
    except ImportError as exc:
        search = RandomizedSearchCV(
            HistGradientBoostingRegressor(
                loss="absolute_error", early_stopping=False, random_state=seed
            ),
            {
                "max_iter": [80, 120, 160],
                "learning_rate": [0.03, 0.05, 0.08],
                "max_leaf_nodes": [7, 15, 23],
                "min_samples_leaf": [20, 40, 60],
                "l2_regularization": [0.5, 2.0, 8.0],
            },
            n_iter=8,
            cv=TimeSeriesSplit(3),
            scoring="neg_mean_absolute_error",
            random_state=seed,
            n_jobs=-1,
        ).fit(X, y)
        return search.best_estimator_, {
            "best_params": search.best_params_,
            "best_cv_mae": -float(search.best_score_),
            "trials": 8,
            "backend": "RandomizedSearchCV fallback",
            "fallback_reason": str(exc),
        }

    def objective(trial):
        m = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=trial.suggest_int("max_iter", 80, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.025, 0.1, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 31),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 80),
            l2_regularization=trial.suggest_float(
                "l2_regularization", 0.1, 20.0, log=True
            ),
            early_stopping=False,
            random_state=seed,
        )
        scores = []
        for a, b in TimeSeriesSplit(n_splits=3).split(X):
            fitted = HistGradientBoostingRegressor(**m.get_params()).fit(
                X.iloc[a], y.iloc[a]
            )
            scores.append(
                regression_metrics(y.iloc[b], fitted.predict(X.iloc[b]))["MAE"]
            )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = HistGradientBoostingRegressor(
        loss="absolute_error",
        **study.best_params,
        early_stopping=False,
        random_state=seed,
    ).fit(X, y)
    return best, {
        "best_params": study.best_params,
        "best_cv_mae": study.best_value,
        "trials": trials,
        "backend": "Optuna",
        "objective": "mean MAE over 3 expanding TimeSeriesSplit folds within training data",
        "seed": seed,
        "n_splits": 3,
        "history": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
