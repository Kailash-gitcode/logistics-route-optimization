"""Delivery-time regression example for the Week 1 logistics project.

Run from the project root: python src/modeling.py

Scikit-learn is used when installed. This workspace does not currently include
it, so a small NumPy implementation of least-squares linear regression and a
randomized tree forest is included as a dependency-light fallback. Both paths
use the same time-based split, features, preprocessing, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
import math
import random
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_logistics_data.csv"
FIGURE_DIR = ROOT / "visualizations"
TARGET = "time_taken_min"
NUMERIC_FEATURES = [
    "approx_delivery_distance_km", "vehicle_condition", "multiple_deliveries",
    "order_hour", "order_day", "delivery_person_age",
]
CATEGORICAL_FEATURES = [
    "road_traffic_density", "weather_conditions", "type_of_vehicle",
    "city", "festival", "day_of_week",
]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset without changing the source file."""
    return pd.read_csv(path, parse_dates=["order_date"], low_memory=False)


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.20):
    """Split on distinct dates so test observations occur after training dates."""
    dated = data.dropna(subset=["order_date"]).copy()
    dates = sorted(dated["order_date"].dt.normalize().unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct valid order dates are required.")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * (1 - test_fraction))))
    cutoff = pd.Timestamp(dates[split_index])
    train = dated[dated["order_date"].dt.normalize() < cutoff].copy()
    test = dated[dated["order_date"].dt.normalize() >= cutoff].copy()
    return train, test, cutoff


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    """Impute from training data and one-hot encode categories consistently.

    Ratings, IDs, raw timestamps, pickup time, and target are deliberately
    excluded. Pickup-related fields would not be available at order placement;
    ratings can be recorded after delivery. Missing numerics use train medians.
    Missing categorical values and unseen test labels use an explicit Unknown
    category. One-hot columns are learned from training data only.
    """
    available_num = [c for c in NUMERIC_FEATURES if c in train and c in test]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in train and c in test]

    x_train_num = train[available_num].apply(pd.to_numeric, errors="coerce")
    x_test_num = test[available_num].apply(pd.to_numeric, errors="coerce")
    medians = x_train_num.median().fillna(0)
    x_train_num = x_train_num.fillna(medians)
    x_test_num = x_test_num.fillna(medians)

    x_train_cat = train[available_cat].astype("string").fillna("Unknown")
    x_test_cat = test[available_cat].astype("string").fillna("Unknown")
    x_train_cat = x_train_cat.replace("", "Unknown")
    x_test_cat = x_test_cat.replace("", "Unknown")
    train_dummies = pd.get_dummies(x_train_cat, dtype=float)
    test_dummies = pd.get_dummies(x_test_cat, dtype=float).reindex(
        columns=train_dummies.columns, fill_value=0.0
    )

    x_train = pd.concat([x_train_num.reset_index(drop=True), train_dummies.reset_index(drop=True)], axis=1)
    x_test = pd.concat([x_test_num.reset_index(drop=True), test_dummies.reset_index(drop=True)], axis=1)
    y_train = pd.to_numeric(train[TARGET], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test[TARGET], errors="coerce").to_numpy(dtype=float)
    return x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float), y_train, y_test, list(x_train.columns)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R-squared without external metric packages."""
    residual = actual - predicted
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - np.sum(residual ** 2) / denominator) if denominator else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


class NumpyLinearRegression:
    """Small ordinary least-squares model used only if scikit-learn is absent."""
    def fit(self, x: np.ndarray, y: np.ndarray):
        design = np.column_stack([np.ones(len(x)), x])
        self.coef_ = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), x]) @ self.coef_


class _TreeNode:
    def __init__(self, value: float):
        self.value = value
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


class NumpyRandomForestRegressor:
    """Educational random forest fallback using bootstrap CART-style trees.

    At each node it samples candidate features and thresholds, selecting the
    split with the largest squared-error reduction. This is intentionally
    compact, not a replacement for a tuned production library implementation.
    """
    def __init__(self, n_estimators: int = 24, max_depth: int = 9,
                 min_samples_leaf: int = 35, max_thresholds: int = 10,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def _grow(self, x, y, depth, rng, importance):
        node = _TreeNode(float(y.mean()))
        n, p = x.shape
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or np.var(y) < 1e-8:
            return node
        feature_count = max(1, int(math.sqrt(p)))
        features = rng.choice(p, size=min(feature_count, p), replace=False)
        parent_sse = float(np.sum((y - y.mean()) ** 2))
        best_gain, best = 0.0, None
        for feature in features:
            values = x[:, feature]
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            if len(unique) > self.max_thresholds + 1:
                quantiles = np.linspace(0.1, 0.9, self.max_thresholds)
                thresholds = np.unique(np.quantile(values, quantiles))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2
            for threshold in thresholds:
                mask = values <= threshold
                left_n = int(mask.sum())
                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                yl, yr = y[mask], y[~mask]
                gain = parent_sse - float(np.sum((yl - yl.mean()) ** 2)) - float(np.sum((yr - yr.mean()) ** 2))
                if gain > best_gain:
                    best_gain, best = gain, (feature, float(threshold), mask)
        if best is None:
            return node
        feature, threshold, mask = best
        node.feature, node.threshold = int(feature), threshold
        importance[feature] += best_gain
        node.left = self._grow(x[mask], y[mask], depth + 1, rng, importance)
        node.right = self._grow(x[~mask], y[~mask], depth + 1, rng, importance)
        return node

    @staticmethod
    def _predict_one(node, row):
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def fit(self, x, y):
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        for _ in range(self.n_estimators):
            sample = rng.integers(0, len(x), size=len(x))
            importance = np.zeros(x.shape[1], dtype=float)
            tree = self._grow(x[sample], y[sample], 0, rng, importance)
            self.trees_.append(tree)
            self.feature_importances_ += importance
        total = self.feature_importances_.sum()
        if total:
            self.feature_importances_ /= total
        return self

    def predict(self, x):
        predictions = np.zeros((len(x), len(self.trees_)), dtype=float)
        for j, tree in enumerate(self.trees_):
            predictions[:, j] = [self._predict_one(tree, row) for row in x]
        return predictions.mean(axis=1)


def save_figures(actual, predictions, metrics, feature_names, importances) -> None:
    """Save actual/predicted, metric comparison, and forest importance figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    navy, blue, gray = "#17324D", "#2878B5", "#D9E0E7"

    # A fixed-seed subset keeps the actual-versus-predicted chart readable.
    rng = np.random.default_rng(42)
    indexes = rng.choice(len(actual), size=min(2500, len(actual)), replace=False)
    lo = float(min(actual.min(), *(p[indexes].min() for p in predictions.values())))
    hi = float(max(actual.max(), *(p[indexes].max() for p in predictions.values())))
    im = Image.new("RGB", (900, 620), "white"); draw = ImageDraw.Draw(im)
    draw.text((35, 20), "Actual versus predicted delivery time (test sample)", fill=navy, font=font)
    for model_name, color, offset in (("Linear Regression", blue, 0), ("Random Forest", "#E07A32", 12)):
        draw.text((45, 52 + offset), model_name, fill=color, font=font)
    left, top, right, bottom = 90, 95, 850, 535
    def xp(v): return left + (float(v) - lo) / max(hi - lo, 1) * (right - left)
    def yp(v): return bottom - (float(v) - lo) / max(hi - lo, 1) * (bottom - top)
    draw.line((left, bottom, right, top), fill=gray, width=2)
    for name, color in (("Linear Regression", blue), ("Random Forest", "#E07A32")):
        for i in indexes:
            draw.ellipse((xp(actual[i])-2, yp(predictions[name][i])-2,
                          xp(actual[i])+2, yp(predictions[name][i])+2), fill=color)
    draw.text((left, bottom + 15), f"{lo:.0f} min", fill=navy, font=font)
    draw.text((right - 40, bottom + 15), f"{hi:.0f} min", fill=navy, font=font)
    draw.text((370, 575), "Actual delivery time (min); diagonal is ideal", fill=navy, font=font)
    im.save(FIGURE_DIR / "regression_actual_vs_predicted.png")

    # Performance chart: MAE/RMSE share units; R2 is shown on its own row.
    im = Image.new("RGB", (900, 430), "white"); draw = ImageDraw.Draw(im)
    draw.text((35, 20), "Test-set model performance comparison", fill=navy, font=font)
    metric_names = ["MAE (min)", "RMSE (min)", "R2"]
    for row, metric in enumerate(metric_names):
        y = 100 + row * 100
        draw.text((45, y), metric, fill=navy, font=font)
        vals = [metrics["Linear Regression"][metric.split()[0]], metrics["Random Forest"][metric.split()[0]]]
        scale = max(max(abs(v) for v in vals), 1e-9)
        for j, (name, color) in enumerate((("Linear Regression", blue), ("Random Forest", "#E07A32"))):
            yy = y + 22 + j * 25; width = 450 * abs(vals[j]) / scale
            draw.rectangle((190, yy, 190 + width, yy + 17), fill=color)
            draw.text((650, yy), f"{name}: {vals[j]:.3f}", fill=navy, font=font)
    im.save(FIGURE_DIR / "regression_model_performance.png")

    # Aggregate one-hot importances to the original human-readable feature.
    aggregated: dict[str, float] = {}
    for name, value in zip(feature_names, importances):
        original = name.split("_", 1)[0] if name not in NUMERIC_FEATURES else name
        # Match known categorical prefixes, including multiword column names.
        for candidate in CATEGORICAL_FEATURES:
            if name.startswith(candidate + "_"):
                original = candidate
                break
        aggregated[original] = aggregated.get(original, 0.0) + float(value)
    top = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)[:12]
    im = Image.new("RGB", (900, 520), "white"); draw = ImageDraw.Draw(im)
    draw.text((35, 20), "Random Forest feature importance (aggregated)", fill=navy, font=font)
    max_value = max((v for _, v in top), default=1)
    for i, (name, value) in enumerate(top):
        y = 75 + i * 34; draw.text((45, y), name, fill=navy, font=font)
        draw.rectangle((230, y, 230 + 560 * value / max_value, y + 20), fill=blue)
        draw.text((800, y), f"{value:.3f}", fill=navy, font=font)
    im.save(FIGURE_DIR / "regression_random_forest_feature_importance.png")


def run_modeling(data_path: str | Path = DATA_PATH, make_figures: bool = True) -> dict[str, Any]:
    """Fit baseline and nonlinear regressors, evaluate, and return artifacts."""
    data = load_data(data_path)
    data = data.dropna(subset=[TARGET])
    train, test, cutoff = chronological_split(data)
    x_train, x_test, y_train, y_test, feature_names = prepare_features(train, test)

    # Use established scikit-learn estimators when present; otherwise use the
    # documented NumPy fallbacks so this educational script remains runnable.
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        linear = LinearRegression()
        forest = RandomForestRegressor(
            n_estimators=120, max_depth=14, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )
        implementation = "scikit-learn"
    except ImportError:
        linear = NumpyLinearRegression()
        forest = NumpyRandomForestRegressor()
        implementation = "NumPy fallback (scikit-learn unavailable)"

    linear.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    predictions = {
        "Linear Regression": linear.predict(x_test),
        "Random Forest": forest.predict(x_test),
    }
    metrics = {name: regression_metrics(y_test, pred) for name, pred in predictions.items()}
    comparison = pd.DataFrame(metrics).T[["MAE", "RMSE", "R2"]]
    better = "Random Forest" if metrics["Random Forest"]["MAE"] < metrics["Linear Regression"]["MAE"] else "Linear Regression"
    importance = pd.Series(forest.feature_importances_, index=feature_names).sort_values(ascending=False)

    if make_figures:
        save_figures(y_test, predictions, metrics, feature_names, forest.feature_importances_)
    return {
        "data": data, "train": train, "test": test, "cutoff": cutoff,
        "feature_names": feature_names, "predictions": predictions,
        "y_test": y_test, "metrics": metrics, "comparison": comparison,
        "better_model": better, "feature_importance": importance,
        "implementation": implementation,
    }


def main() -> None:
    results = run_modeling()
    print(f"Implementation: {results['implementation']}")
    print(f"Chronological split cutoff: {results['cutoff'].date()}")
    print(f"Train/test rows: {len(results['train']):,} / {len(results['test']):,}")
    print("\nTest-set metrics (minutes for MAE/RMSE; unitless R2):")
    print(results["comparison"].round(3).to_string())
    print(f"\nBetter model by lower MAE: {results['better_model']}")
    print("\nTop Random Forest features:")
    print(results["feature_importance"].head(12).round(4).to_string())
    print(f"\nFigures saved in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
