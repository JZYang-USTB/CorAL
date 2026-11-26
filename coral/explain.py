import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def analyze_shap(compositions, efficiency, features, plot_path=None):
    import pandas as pd
    import shap

    x = pd.DataFrame(compositions, columns=features, dtype=float)
    y = np.asarray(efficiency, dtype=float)
    if len(x) < 10 or not (np.isfinite(x.to_numpy()).all() and np.isfinite(y).all()):
        raise ValueError("SHAP requires at least ten finite observations.")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=36)
    model = RandomForestRegressor(random_state=100, n_jobs=1).fit(train_x, train_y)
    explanation = shap.Explainer(model)(train_x)
    direction = {}
    for index, feature in enumerate(features):
        values = explanation.data[:, index]
        low, high = np.quantile(values, [0.25, 0.75])
        if low < high:
            direction[feature] = {
                "low_cutoff_pct": float(low),
                "high_cutoff_pct": float(high),
                "low_mean_shap": float(explanation.values[values <= low, index].mean()),
                "high_mean_shap": float(explanation.values[values >= high, index].mean()),
            }
    if plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.plots.beeswarm(explanation, show=False)
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close()
    return {
        "feature_order": list(features),
        "train_rows": train_x.index.tolist(),
        "test_rows": test_x.index.tolist(),
        "test_rmse": float(np.sqrt(mean_squared_error(test_y, model.predict(test_x)))),
        "mean_absolute_shap": dict(zip(features, np.abs(explanation.values).mean(axis=0).tolist(), strict=True)),
        "low_and_high_fraction_mean_shap": direction,
        "feature_values": explanation.data.tolist(),
        "shap_values": explanation.values.tolist(),
        "base_values": np.asarray(explanation.base_values).tolist(),
    }
