"""Exploratory geographic clustering of delivery locations.

Run from the project root: python src/clustering.py

Clusters are exploratory patterns in delivery coordinates, not official
dispatch zones. DBSCAN uses Haversine distance when scikit-learn is available.
The SciPy fallback searches a 3-D unit sphere using the equivalent chord radius.
"""

from __future__ import annotations

from pathlib import Path
import colorsys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_logistics_data.csv"
FIGURE_DIR = ROOT / "visualizations"
EARTH_RADIUS_KM = 6371.0088
DEFAULT_EPS_KM = 1.0
DEFAULT_MIN_SAMPLES = 20


def load_valid_deliveries(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load only rows with both coordinate pairs explicitly validated.

    The additional numeric checks guard against zero, placeholder, and
    out-of-region values even if an input was not prepared by our script.
    """
    data = pd.read_csv(path, low_memory=False)
    needed = [
        "restaurant_coordinates_valid", "delivery_coordinates_valid",
        "restaurant_latitude", "restaurant_longitude",
        "delivery_location_latitude", "delivery_location_longitude",
    ]
    missing = [column for column in needed if column not in data]
    if missing:
        raise ValueError(f"Required validated coordinate fields are missing: {missing}")
    valid = data["restaurant_coordinates_valid"].eq(True) & data[
        "delivery_coordinates_valid"
    ].eq(True)
    coordinate_columns = needed[2:]
    for column in coordinate_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        valid &= data[column].notna() & data[column].ne(0)
    # Broad bounds match this India-focused dataset and reject placeholders.
    valid &= data["restaurant_latitude"].between(6, 38)
    valid &= data["delivery_location_latitude"].between(6, 38)
    valid &= data["restaurant_longitude"].between(68, 98)
    valid &= data["delivery_location_longitude"].between(68, 98)
    return data.loc[valid].copy().reset_index(drop=True)


def _unit_sphere_coordinates(latitude_longitude_radians: np.ndarray) -> np.ndarray:
    """Map lat/lon to 3-D unit-sphere points for exact radius searches."""
    lat, lon = latitude_longitude_radians[:, 0], latitude_longitude_radians[:, 1]
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def scipy_dbscan_haversine(
    coordinates_radians: np.ndarray,
    eps_radians: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN using SciPy cKDTree and a Haversine-equivalent chord radius.

    For angular Haversine distance eps, two unit-sphere points are neighbors
    exactly when their Euclidean chord distance is <= 2*sin(eps/2). This avoids
    treating latitude and longitude as ordinary Cartesian coordinates.
    """
    sphere_points = _unit_sphere_coordinates(coordinates_radians)
    chord_radius = 2.0 * np.sin(eps_radians / 2.0)
    tree = cKDTree(sphere_points)
    labels = np.full(len(sphere_points), -2, dtype=int)  # -2 = not visited
    cluster_id = 0

    for point_index in range(len(sphere_points)):
        if labels[point_index] != -2:
            continue
        neighbors = tree.query_ball_point(sphere_points[point_index], chord_radius)
        if len(neighbors) < min_samples:
            labels[point_index] = -1  # noise; may later be assigned as a border point
            continue

        labels[point_index] = cluster_id
        queue = list(neighbors)
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            candidate = queue[cursor]
            cursor += 1
            if labels[candidate] == -1:
                labels[candidate] = cluster_id
            if labels[candidate] != -2:
                continue
            labels[candidate] = cluster_id
            candidate_neighbors = tree.query_ball_point(sphere_points[candidate], chord_radius)
            if len(candidate_neighbors) >= min_samples:
                for neighbor in candidate_neighbors:
                    if neighbor not in queued:
                        queue.append(neighbor)
                        queued.add(neighbor)
        cluster_id += 1
    return labels


def fit_dbscan(
    deliveries: pd.DataFrame,
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> tuple[np.ndarray, str]:
    """Fit Haversine DBSCAN, choosing scikit-learn or equivalent SciPy search."""
    coords = np.radians(
        deliveries[["delivery_location_latitude", "delivery_location_longitude"]]
        .to_numpy(dtype=float)
    )
    eps_radians = eps_km / EARTH_RADIUS_KM
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(
            eps=eps_radians, min_samples=min_samples, metric="haversine", algorithm="ball_tree"
        ).fit_predict(coords)
        return labels, "scikit-learn DBSCAN (Haversine metric)"
    except ImportError:
        labels = scipy_dbscan_haversine(coords, eps_radians, min_samples)
        return labels, "SciPy cKDTree DBSCAN (Haversine-equivalent spherical chord distance)"


def cluster_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize size, delivery time, and approximate distance per cluster."""
    clustered = data[data["cluster"] >= 0]
    summary = clustered.groupby("cluster").agg(
        records=("cluster", "size"),
        average_delivery_time_min=("time_taken_min", "mean"),
        median_delivery_time_min=("time_taken_min", "median"),
        average_approx_distance_km=("approx_delivery_distance_km", "mean"),
        median_latitude=("delivery_location_latitude", "median"),
        median_longitude=("delivery_location_longitude", "median"),
    )
    summary["share_of_valid_records_pct"] = summary["records"] / len(data) * 100
    return summary.sort_values("records", ascending=False).round(2)


def _cluster_color(cluster: int) -> tuple[int, int, int]:
    if cluster == -1:
        return (155, 160, 166)
    hue = (cluster * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 0.78)
    return tuple(int(channel * 255) for channel in rgb)


def save_visualizations(data: pd.DataFrame, summary: pd.DataFrame, out_dir: str | Path = FIGURE_DIR) -> None:
    """Write geographic clusters and cluster-level comparison charts as PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    navy, grid = (23, 50, 77), (220, 225, 230)

    # Geographic plot: coordinates remain degrees for display only; clustering
    # itself was performed with spherical geographic distances.
    width, height = 1150, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 20), "Delivery locations colored by exploratory DBSCAN cluster", fill=navy, font=font)
    left, top, right, bottom = 85, 75, width - 35, height - 90
    lon_min, lon_max = float(data.delivery_location_longitude.min()), float(data.delivery_location_longitude.max())
    lat_min, lat_max = float(data.delivery_location_latitude.min()), float(data.delivery_location_latitude.max())
    for tick in range(6):
        x = left + tick * (right - left) / 5
        y = top + tick * (bottom - top) / 5
        draw.line((x, top, x, bottom), fill=grid)
        draw.line((left, y, right, y), fill=grid)
        draw.text((x - 12, bottom + 9), f"{lon_min + tick*(lon_max-lon_min)/5:.1f}", fill=navy, font=font)
        draw.text((35, y - 5), f"{lat_max - tick*(lat_max-lat_min)/5:.1f}", fill=navy, font=font)
    x_values = left + (data.delivery_location_longitude - lon_min) / max(lon_max - lon_min, 1e-9) * (right - left)
    y_values = bottom - (data.delivery_location_latitude - lat_min) / max(lat_max - lat_min, 1e-9) * (bottom - top)
    for x, y, cluster in zip(x_values, y_values, data.cluster):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=_cluster_color(int(cluster)))
    draw.text((width // 2 - 45, height - 25), "Longitude", fill=navy, font=font)
    draw.text((width - 135, 50), "Gray = noise", fill=_cluster_color(-1), font=font)
    image.save(out_dir / "delivery_location_dbscan_clusters.png")

    # Cluster size comparison includes every cluster, sorted largest first.
    def bar_chart(values, title, xlabel, filename, value_format="{:.0f}"):
        # Keep charts readable. The full cluster table remains available in the notebook.
        values = values.head(20)
        chart = Image.new("RGB", (1150, max(430, 100 + 32 * len(values))), "white")
        pen = ImageDraw.Draw(chart)
        pen.text((45, 20), title, fill=navy, font=font)
        if values.empty:
            pen.text((60, 80), "No clusters were found for this configuration.", fill=navy, font=font)
        vmax = max(float(values.max()) if len(values) else 0.0, 1e-9)
        for index, (label, value) in enumerate(values.items()):
            y = 70 + index * 32
            pen.text((45, y), f"Cluster {label}", fill=navy, font=font)
            x0 = 155; x1 = x0 + 760 * float(value) / vmax
            pen.rectangle((x0, y, x1, y + 19), fill=(40, 120, 181))
            pen.text((x1 + 8, y), value_format.format(value), fill=navy, font=font)
        pen.text((430, chart.height - 28), xlabel, fill=navy, font=font)
        chart.save(out_dir / filename)

    largest_clusters = summary.records.head(20).index
    bar_chart(summary.records, "Cluster size comparison (20 largest clusters)", "Valid delivery records", "delivery_cluster_sizes.png")
    bar_chart(
        summary.loc[largest_clusters, "average_delivery_time_min"],
        "Average delivery time (20 largest clusters)",
        "Average observed delivery time (minutes)",
        "delivery_cluster_average_time.png",
        "{:.1f} min",
    )
    bar_chart(
        summary.loc[largest_clusters, "average_approx_distance_km"],
        "Average distance (20 largest clusters)",
        "Straight-line approximate distance (km)",
        "delivery_cluster_average_distance.png",
        "{:.1f} km",
    )


def run_clustering(
    data_path: str | Path = DATA_PATH,
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Load validated data, cluster locations, summarize, and plot results."""
    deliveries = load_valid_deliveries(data_path)
    if deliveries.empty:
        raise ValueError("No records have validated restaurant and delivery coordinates.")
    labels, implementation = fit_dbscan(deliveries, eps_km=eps_km, min_samples=min_samples)
    deliveries["cluster"] = labels
    summary = cluster_summary(deliveries)
    cluster_count = int(summary.shape[0])
    noise_count = int((labels == -1).sum())
    noise_percent = noise_count / len(labels) * 100
    if make_figures:
        save_visualizations(deliveries, summary)
    return {
        "deliveries": deliveries, "summary": summary,
        "valid_records": len(deliveries), "cluster_count": cluster_count,
        "noise_count": noise_count, "noise_percent": noise_percent,
        "implementation": implementation, "eps_km": eps_km,
        "min_samples": min_samples,
    }


def main() -> None:
    results = run_clustering()
    print("Implementation:", results["implementation"])
    print(f"DBSCAN settings: eps={results['eps_km']:.2f} km, min_samples={results['min_samples']}")
    print(f"Valid geographic records: {results['valid_records']:,}")
    print(f"Clusters found: {results['cluster_count']}")
    print(f"Noise/outlier records: {results['noise_count']:,} ({results['noise_percent']:.2f}%)")
    print("Cluster-level summary:")
    print(results["summary"].to_string())
    print(f"Figures saved in {FIGURE_DIR}")
    print("These clusters are exploratory geographic patterns, not official dispatch zones.")


if __name__ == "__main__":
    main()
