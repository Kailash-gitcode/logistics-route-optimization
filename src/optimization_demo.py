"""Small educational batch-assignment optimization for food delivery.

Run from the project root: python src/optimization_demo.py

This is an illustrative workload-balancing exercise, not a vehicle-routing
system. The demo uses historical delivery durations as hypothetical workload
estimates. It does not recommend real routes or dispatch decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_logistics_data.csv"
FIGURE_DIR = ROOT / "visualizations"
DEMO_ORDERS = 12
BATCH_CAPACITY = 4
NUM_BATCHES = 3
GRID_DEGREES = 0.1
RANDOM_STATE = 42


def load_demo_orders(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Pick 12 real deliveries from the most populated small geographic tile.

    Both coordinate pairs must have passed preprocessing validation. Selecting
    a dense 0.1-degree destination tile makes this a local-area illustration;
    the tile is only a sampling convenience, not an operational service zone.
    """
    data = pd.read_csv(path, low_memory=False)
    required = [
        "id", "restaurant_coordinates_valid", "delivery_coordinates_valid",
        "restaurant_latitude", "restaurant_longitude", "delivery_location_latitude",
        "delivery_location_longitude", "approx_delivery_distance_km",
        "time_taken_min", "multiple_deliveries",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns are missing from the cleaned file: {missing}")

    valid = data["restaurant_coordinates_valid"].eq(True) & data[
        "delivery_coordinates_valid"
    ].eq(True)
    for column in [
        "restaurant_latitude", "restaurant_longitude",
        "delivery_location_latitude", "delivery_location_longitude",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        valid &= data[column].notna() & data[column].ne(0)
    valid &= data["restaurant_latitude"].between(6, 38)
    valid &= data["delivery_location_latitude"].between(6, 38)
    valid &= data["restaurant_longitude"].between(68, 98)
    valid &= data["delivery_location_longitude"].between(68, 98)
    valid &= pd.to_numeric(data["time_taken_min"], errors="coerce").notna()
    valid &= pd.to_numeric(data["approx_delivery_distance_km"], errors="coerce").notna()
    candidates = data.loc[valid].copy()
    if len(candidates) < DEMO_ORDERS:
        raise ValueError(f"At least {DEMO_ORDERS} valid deliveries are required.")

    # The busiest rounded 0.1-degree tile provides a small, geographically
    # concentrated sample. It is not asserted to be an official service area.
    candidates["_latitude_tile"] = (candidates["delivery_location_latitude"] / GRID_DEGREES).round().astype(int)
    candidates["_longitude_tile"] = (candidates["delivery_location_longitude"] / GRID_DEGREES).round().astype(int)
    tile_counts = candidates.groupby(["_latitude_tile", "_longitude_tile"], sort=True).size()
    busiest_tile = tile_counts.sort_values(ascending=False).index[0]
    tile = candidates[
        candidates["_latitude_tile"].eq(busiest_tile[0])
        & candidates["_longitude_tile"].eq(busiest_tile[1])
    ]
    sample = tile.sample(n=DEMO_ORDERS, random_state=RANDOM_STATE).copy().reset_index(drop=True)

    # The source's observed duration is used as a hypothetical workload proxy.
    # It is not a prospective dispatch estimate and contains hindsight.
    sample["estimated_duration_min"] = pd.to_numeric(sample["time_taken_min"], errors="coerce")
    sample["approx_delivery_distance_km"] = pd.to_numeric(
        sample["approx_delivery_distance_km"], errors="coerce"
    )
    sample["multiple_deliveries"] = pd.to_numeric(sample["multiple_deliveries"], errors="coerce")
    return sample.drop(columns=["_latitude_tile", "_longitude_tile"])


def assign_baseline(orders: pd.DataFrame) -> list[int]:
    """Sequentially assign input rows to batches, up to batch capacity."""
    return [index // BATCH_CAPACITY for index in range(len(orders))]


def optimize_assignments(orders: pd.DataFrame) -> list[int]:
    """Exactly enumerate feasible assignments and minimize maximum batch load.

    Constraint: every delivery is assigned once; there are exactly three
    batches, each with at most four deliveries. The objective is to minimize
    makespan: the largest sum of estimated task durations on any batch. Equal
    makespan solutions use squared load spread as a deterministic tie-breaker.
    """
    durations = orders["estimated_duration_min"].to_numpy(dtype=float)
    n = len(durations)
    if n != NUM_BATCHES * BATCH_CAPACITY:
        raise ValueError(
            f"Exact demo expects {NUM_BATCHES * BATCH_CAPACITY} orders, got {n}."
        )

    best_key = (float("inf"), float("inf"))
    best_assignment: list[int] | None = None
    assignment = [-1] * n
    loads = [0.0] * NUM_BATCHES
    counts = [0] * NUM_BATCHES

    def search(order_index: int, active_batches: int) -> None:
        nonlocal best_key, best_assignment
        if order_index == n:
            if active_batches != NUM_BATCHES:
                return
            key = (max(loads), sum(load * load for load in loads))
            if key < best_key:
                best_key = key
                best_assignment = assignment.copy()
            return

        # Remove symmetric batch permutations: only open the next new batch.
        last_option = min(active_batches, NUM_BATCHES - 1)
        for batch in range(last_option + 1):
            if counts[batch] >= BATCH_CAPACITY:
                continue
            was_new = batch == active_batches
            assignment[order_index] = batch
            counts[batch] += 1
            loads[batch] += durations[order_index]

            # Positive durations mean a partial load above the best makespan
            # cannot lead to a better solution.
            if max(loads) <= best_key[0]:
                search(order_index + 1, active_batches + int(was_new))

            loads[batch] -= durations[order_index]
            counts[batch] -= 1
            assignment[order_index] = -1

    search(0, 0)
    if best_assignment is None:
        raise RuntimeError("No feasible assignment was found.")
    return best_assignment


def batch_loads(orders: pd.DataFrame, assignment: list[int]) -> list[float]:
    """Sum hypothetical workload durations for each batch."""
    return [
        float(orders.loc[np.asarray(assignment) == batch, "estimated_duration_min"].sum())
        for batch in range(NUM_BATCHES)
    ]


def save_figures(orders, baseline, optimized, baseline_loads, optimized_loads) -> None:
    """Save a makespan comparison and a per-delivery batch assignment chart."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    navy, blue, orange, gray = "#17324D", "#2878B5", "#E07A32", "#D9E0E7"

    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 24), "Baseline versus optimized completion-time proxy", fill=navy, font=font)
    plot_left, plot_right, plot_top, plot_bottom = 100, 830, 100, 390
    values = [max(baseline_loads), max(optimized_loads)]
    maximum = max(values) * 1.15
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=navy, width=2)
    for i, (label, value, color) in enumerate(zip(["Baseline", "Optimized"], values, [blue, orange])):
        x = 260 + i * 300
        y = plot_bottom - value / maximum * (plot_bottom - plot_top)
        draw.rectangle((x, y, x + 145, plot_bottom), fill=color)
        draw.text((x + 25, y - 20), f"{value:.1f} min", fill=navy, font=font)
        draw.text((x + 35, plot_bottom + 15), label, fill=navy, font=font)
    draw.text((320, 450), "Makespan = largest batch workload (minutes)", fill=navy, font=font)
    image.save(FIGURE_DIR / "optimization_makespan_comparison.png")

    # Rows show each selected delivery's estimated workload and assigned batch.
    image = Image.new("RGB", (1100, 620), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 20), "Illustrative delivery-to-batch assignments", fill=navy, font=font)
    draw.text((75, 55), "Baseline: sequential", fill=blue, font=font)
    draw.text((600, 55), "Optimized: balanced workload", fill=orange, font=font)
    col_x = [80, 605]
    colors = ["#2878B5", "#E07A32", "#4D9B67"]
    for i, row in orders.iterrows():
        y = 90 + i * 39
        duration = float(row.estimated_duration_min)
        dist = float(row.approx_delivery_distance_km)
        for x0, assignment in zip(col_x, [baseline, optimized]):
            batch = int(assignment[i])
            draw.rectangle((x0, y, x0 + 28, y + 24), fill=colors[batch])
            draw.text((x0 + 38, y + 5), f"Order {i+1}: {duration:.0f} min, {dist:.1f} km", fill=navy, font=font)
            draw.text((x0 + 365, y + 5), f"Batch {batch+1}", fill=navy, font=font)
    draw.text((65, 575), "Colors identify batches. Distance is point-to-point approximation, not a route cost.", fill=navy, font=font)
    image.save(FIGURE_DIR / "optimization_batch_assignments.png")


def run_demo(data_path: str | Path = DATA_PATH, make_figures: bool = True) -> dict[str, Any]:
    """Run the complete assignment comparison and return metrics and tables."""
    orders = load_demo_orders(data_path)
    baseline = assign_baseline(orders)
    optimized = optimize_assignments(orders)
    baseline_load = batch_loads(orders, baseline)
    optimized_load = batch_loads(orders, optimized)
    baseline_makespan = max(baseline_load)
    optimized_makespan = max(optimized_load)
    improvement = baseline_makespan - optimized_makespan
    improvement_pct = improvement / baseline_makespan * 100 if baseline_makespan else 0.0
    orders["baseline_batch"] = np.asarray(baseline) + 1
    orders["optimized_batch"] = np.asarray(optimized) + 1
    orders["estimated_duration_min"] = orders["estimated_duration_min"].astype(float)
    if make_figures:
        save_figures(orders, baseline, optimized, baseline_load, optimized_load)
    return {
        "orders": orders,
        "baseline_assignment": baseline,
        "optimized_assignment": optimized,
        "baseline_batch_loads": baseline_load,
        "optimized_batch_loads": optimized_load,
        "baseline_makespan_min": baseline_makespan,
        "optimized_makespan_min": optimized_makespan,
        "improvement_min": improvement,
        "improvement_percent": improvement_pct,
        "total_estimated_work_min": float(orders.estimated_duration_min.sum()),
        "number_of_deliveries": len(orders),
        "number_of_batches": NUM_BATCHES,
        "capacity_per_batch": BATCH_CAPACITY,
    }


def main() -> None:
    results = run_demo()
    print("Educational batch assignment demonstration (not a route recommendation)")
    print(f"Deliveries: {results['number_of_deliveries']}; batches/vehicles: {results['number_of_batches']}; capacity: {results['capacity_per_batch']} deliveries per batch")
    print(f"Baseline estimated completion-time proxy (makespan): {results['baseline_makespan_min']:.1f} min")
    print(f"Optimized estimated completion-time proxy (makespan): {results['optimized_makespan_min']:.1f} min")
    print(f"Improvement: {results['improvement_min']:.1f} min ({results['improvement_percent']:.1f}%)")
    print(f"Total estimated work across individual deliveries (unchanged): {results['total_estimated_work_min']:.1f} min")
    print("Baseline batch loads:", [round(value, 1) for value in results["baseline_batch_loads"]])
    print("Optimized batch loads:", [round(value, 1) for value in results["optimized_batch_loads"]])
    print("Figures saved in", FIGURE_DIR)


if __name__ == "__main__":
    main()
