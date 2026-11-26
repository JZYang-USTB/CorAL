from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample


FOREST_GRID = {
    "n_estimators": [10, 100],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10],
}


def composition_grid():
    dividers = np.array(list(combinations(range(23), 3)), dtype=int)
    bounds = np.column_stack((np.full(len(dividers), -1), dividers, np.full(len(dividers), 23)))
    return 5 * (np.diff(bounds, axis=1) - 1)


def expected_improvement(mean, deviation, best):
    gain = np.asarray(mean, dtype=float) - best
    deviation = np.asarray(deviation, dtype=float)
    z = np.divide(gain, deviation, out=np.zeros_like(gain), where=deviation > 0)
    ei = gain * norm.cdf(z) + deviation * norm.pdf(z)
    return np.maximum(np.where(deviation > 0, ei, gain), 0)


def _bootstrap_prediction(x, y, space, seed):
    sampled_x, sampled_y = resample(x, y, random_state=seed)
    search = GridSearchCV(
        RandomForestRegressor(random_state=0, n_jobs=1), FOREST_GRID,
        cv=10, scoring="neg_mean_squared_error", error_score="raise",
    )
    search.fit(sampled_x, sampled_y)
    return search.predict(space), search.best_params_


def rank_compositions(compositions, efficiency, *, excluded=(), workers=1, seed=0, bootstraps=50):
    x = np.asarray(compositions, dtype=float)
    y = np.asarray(efficiency, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x),) or len(x) < 30:
        raise ValueError("At least 30 observed four-component formulations are required.")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and (x >= 0).all()
            and np.allclose(x.sum(axis=1), 100, rtol=0) and (y <= 100).all()):
        raise ValueError("Use finite nonnegative compositions totaling 100 and finite IE values at most 100.")
    if not 1 <= workers or not 1 <= bootstraps:
        raise ValueError("Workers and bootstrap count must be positive.")
    space = composition_grid()
    evaluated = {tuple(np.round(row, 8)) for row in x}
    evaluated.update(tuple(np.round(row, 8)) for row in excluded)
    remaining = [i for i, row in enumerate(space) if tuple(row) not in evaluated]
    if not remaining:
        raise ValueError("The 5 percent composition grid is exhausted.")
    predictions = Parallel(n_jobs=workers)(
        delayed(_bootstrap_prediction)(x, y, space, seed + index)
        for index in range(bootstraps)
    )
    values = np.stack([prediction for prediction, _ in predictions])
    mean, std = values.mean(axis=0), values.std(axis=0)
    best = float(y.max())
    acquisition = expected_improvement(mean, std, best)
    ranking = sorted(remaining, key=lambda i: (-acquisition[i], -mean[i], tuple(space[i])))
    candidates = [
        {"composition_pct": space[i].tolist(), "mean_ie_pct": float(mean[i]),
         "sd_ie_pct": float(std[i]), "expected_improvement_pct": float(acquisition[i])}
        for i in ranking[:5]
    ]
    return {
        "observations": len(x), "grid_size": len(space), "unmeasured_grid_size": len(remaining),
        "best_observed_ie_pct": best, "bootstrap_models": bootstraps, "cv_folds": 10,
        "bootstrap_seed_start": seed, "best_parameters": [params for _, params in predictions],
        "proposals": candidates,
    }
