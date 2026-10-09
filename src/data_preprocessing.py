"""Transparent, reproducible preprocessing and quality audit for delivery data.

Run from the project root:
    python src/data_preprocessing.py

By default, this reads the original CSV without modifying it and writes a
separate Week 2 result to data/week2_cleaned_logistics_data.csv. To choose a
different output path, pass --output PATH. The Week 1 cleaned file is never
overwritten by default.

Cleaning policies are conservative: source records are preserved, duplicates
and statistical outliers are flagged rather than dropped, suspicious
coordinates remain visible but are excluded from distance calculation, and
unverified category or time values are not silently corrected.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "Zomato Dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "week2_cleaned_logistics_data.csv"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "reports" / "week2_data_quality_summary.json"

EARTH_RADIUS_KM = 6371.0088

# These are the expected values observed in this source file, not universal
# business rules. Values outside the list are reported and kept for review.
CATEGORY_EXPECTATIONS: dict[str, tuple[str, ...]] = {
    "weather_conditions": ("Fog", "Stormy", "Cloudy", "Sandstorms", "Windy", "Sunny"),
    "road_traffic_density": ("Low", "Medium", "High", "Jam"),
    "type_of_order": ("Snack", "Meal", "Drinks", "Buffet"),
    "type_of_vehicle": ("motorcycle", "scooter", "electric_scooter", "bicycle"),
    "festival": ("Yes", "No"),
    # Preserve the dataset spelling. "Metropolitian" is not silently corrected.
    "city": ("Metropolitian", "Urban", "Semi-Urban"),
}

# Raw numeric fields; numeric-looking time strings are not measurements.
NUMERIC_SOURCE_COLUMNS = {
    "delivery_person_age", "delivery_person_ratings", "restaurant_latitude",
    "restaurant_longitude", "delivery_location_latitude",
    "delivery_location_longitude", "vehicle_condition", "multiple_deliveries",
    "time_taken_min",
}

# Broad bounds are an explicit India-coverage assumption for this dataset.
# They are separate from globally valid latitude/longitude ranges.
EXPECTED_INDIA_BOUNDS = {
    "latitude": (6.0, 38.0),
    "longitude": (68.0, 98.0),
}

# Field descriptions are practical interpretations of the column names. The
# dataset has no verified publisher data dictionary, so ambiguous meanings
# (especially multiple_deliveries and vehicle_condition) must be confirmed.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "id": "Order identifier; not a predictive feature.",
    "delivery_person_id": "Delivery-person identifier; definition/source is undocumented.",
    "delivery_person_age": "Recorded delivery-person age; unit presumed years.",
    "delivery_person_ratings": "Recorded delivery-person rating; a 1–5 range is tentative and requires source confirmation.",
    "restaurant_latitude": "Restaurant latitude in decimal degrees.",
    "restaurant_longitude": "Restaurant longitude in decimal degrees.",
    "delivery_location_latitude": "Delivery destination latitude in decimal degrees.",
    "delivery_location_longitude": "Delivery destination longitude in decimal degrees.",
    "order_date": "Order date as supplied; source format is day-month-year.",
    "time_orderd": "Order time as supplied; source column spelling is retained.",
    "time_order_picked": "Pickup time as supplied; no pickup date or timezone is present.",
    "weather_conditions": "Weather category; collection method and location specificity are undocumented.",
    "road_traffic_density": "Traffic category; measurement method and route specificity are undocumented.",
    "vehicle_condition": "Recorded integer condition code; code meanings are undocumented.",
    "type_of_order": "Order/food category.",
    "type_of_vehicle": "Vehicle category.",
    "multiple_deliveries": "Numeric delivery/batching field; exact count semantics require confirmation.",
    "festival": "Festival indicator/category; event definition is undocumented.",
    "city": "Broad area/city-type label, not a reliable city name.",
    "time_taken_min": "Recorded delivery duration in minutes; target in Week 1 modeling.",
}


def load_dataset(csv_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load a CSV without modifying it or coercing mixed-format time fields."""
    return pd.read_csv(csv_path, low_memory=False)


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case names and reject accidental collisions."""
    result = data.copy()
    names = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()))
        .strip("_")
        .lower()
        for name in result.columns
    ]
    duplicates = pd.Index(names)[pd.Index(names).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Column-name normalization would create duplicates: {duplicates}")
    result.columns = names
    return result


def parse_order_date(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a parsed date and status while retaining the original date text."""
    result = data.copy()
    source = "order_date"
    raw = result[source] if source in result else pd.Series(pd.NA, index=result.index)
    parsed = pd.to_datetime(raw, format="%d-%m-%Y", errors="coerce")
    result["order_date_parsed"] = parsed
    source_missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
    failed = ~source_missing & parsed.isna()
    result["order_date_parse_status"] = np.select(
        [source_missing, failed], ["source_missing", "parse_failed"], default="parsed"
    )
    diagnostics = {
        "source_column": source,
        "source_missing_count": int(source_missing.sum()),
        "parse_failure_count": int(failed.sum()),
        "failure_examples": _failure_examples(result, source, failed),
    }
    return result, diagnostics


def parse_time_to_minutes(value: object) -> float:
    """Parse HH:MM[:SS] or an unambiguous fractional day; otherwise return NaN.

    Numeric strings in [0, 1) are treated as spreadsheet fractions of a day.
    The string "1" is not interpreted as midnight, and times such as 24:05:00
    are not repaired or rolled over because the source convention is unknown.
    """
    if pd.isna(value):
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan

    try:
        numeric_value = float(text)
        if 0 <= numeric_value < 1:
            return numeric_value * 24 * 60
        return math.nan
    except (TypeError, ValueError):
        pass

    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return math.nan
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return math.nan
    return hour * 60 + minute + second / 60


def parse_order_and_pickup_times(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Add parsed time fields/status; preserve raw values and report failures.

    The optional order-to-pickup interval retains the prior project's heuristic:
    if pickup clock time is earlier than order clock time, add 24 hours, then
    reject intervals above 12 hours. This is an assumption, not source-verified:
    there is no pickup date, and a value like 24:05:00 is left unparsed.
    """
    result = data.copy()
    diagnostics: dict[str, dict[str, Any]] = {}
    for source, target in (
        ("time_orderd", "order_time_minutes"),
        ("time_order_picked", "pickup_time_minutes"),
    ):
        raw = result[source] if source in result else pd.Series(pd.NA, index=result.index)
        parsed = raw.map(parse_time_to_minutes).astype(float)
        source_missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
        failed = ~source_missing & parsed.isna()
        result[target] = parsed
        result[f"{target}_parse_status"] = np.select(
            [source_missing, failed], ["source_missing", "parse_failed"], default="parsed"
        )
        diagnostics[source] = {
            "parsed_column": target,
            "source_missing_count": int(source_missing.sum()),
            "parse_failure_count": int(failed.sum()),
            "failure_examples": _failure_examples(result, source, failed),
        }

    interval = result["pickup_time_minutes"] - result["order_time_minutes"]
    crosses_midnight_assumed = interval < 0
    interval = interval.where(~crosses_midnight_assumed, interval + 24 * 60)
    interval_over_limit = interval > 12 * 60
    result["pickup_crosses_midnight_assumed"] = crosses_midnight_assumed.astype("boolean")
    result["order_to_pickup_over_12h_flag"] = interval_over_limit.astype("boolean")
    result["order_to_pickup_minutes"] = interval.mask(interval_over_limit)
    result["order_hour"] = (result["order_time_minutes"] // 60).astype("Int64")
    result["pickup_hour"] = (result["pickup_time_minutes"] // 60).astype("Int64")
    diagnostics["order_to_pickup_interval_assumption"] = {
        "negative_clock_differences_rolled_forward": int(crosses_midnight_assumed.sum()),
        "intervals_over_12_hours_set_missing": int(interval_over_limit.sum()),
        "note": "Heuristic only; source has no pickup date, and 24:xx values are not normalized.",
    }
    return result, diagnostics


def _failure_examples(data: pd.DataFrame, source: str, mask: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    """Return raw parse-failure examples with row numbers and IDs where present."""
    columns = [source] if source in data else []
    if "id" in data:
        columns.insert(0, "id")
    examples = data.loc[mask, columns].head(limit).copy()
    examples.insert(0, "source_row_index", examples.index)
    return examples.astype(object).where(pd.notna(examples), None).to_dict("records")


def normalize_categories(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Trim known category fields and canonicalize case only for known labels.

    Unknown nonblank labels remain trimmed but otherwise unchanged. Missing and
    blank category values become the explicit label "Unknown". Source spellings
    such as "Metropolitian" are preserved because no authoritative correction
    is available. Returns counts of values whose whitespace/case was normalized.
    """
    result = data.copy()
    changed: dict[str, int] = {}
    for column, expected in CATEGORY_EXPECTATIONS.items():
        if column not in result:
            continue
        lookup = {label.casefold(): label for label in expected}
        source = result[column].astype("string")
        trimmed = source.str.strip()
        canonical = trimmed.map(lambda value: lookup.get(value.casefold(), value) if pd.notna(value) else value)
        canonical = canonical.fillna("Unknown").replace("", "Unknown")
        changed[column] = int((source.notna() & source.str.strip().ne("") & source.ne(canonical)).sum())
        result[column] = canonical
    return result, changed


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative missing handling without imputing numeric measures.

    Categoricals use "Unknown"; numeric measures remain missing. Source missing
    counts are retained separately by the audit. No rows are deleted.
    """
    result, _ = normalize_categories(data)
    return result


def validate_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Add global, zero, and India-region flags without altering source values.

    Negative latitude/longitude values can be globally valid. They are flagged
    as outside the expected India region here, not rejected merely for sign.
    Original coordinate values remain in the output. Distance is allowed only
    when all four values are finite, globally in range, nonzero, and within the
    explicitly assumed broad India bounds.
    """
    result = data.copy()
    pair_defs = {
        "restaurant": ("restaurant_latitude", "restaurant_longitude"),
        "delivery": ("delivery_location_latitude", "delivery_location_longitude"),
    }
    for prefix, (lat_column, lon_column) in pair_defs.items():
        lat = pd.to_numeric(result[lat_column], errors="coerce")
        lon = pd.to_numeric(result[lon_column], errors="coerce")
        finite = np.isfinite(lat) & np.isfinite(lon)
        global_range = lat.between(-90, 90) & lon.between(-180, 180)
        zero = lat.eq(0) | lon.eq(0)
        near_zero = lat.abs().lt(1) | lon.abs().lt(1)
        india_region = lat.between(*EXPECTED_INDIA_BOUNDS["latitude"]) & lon.between(
            *EXPECTED_INDIA_BOUNDS["longitude"]
        )

        result[f"{prefix}_coordinates_finite"] = finite.astype(bool)
        result[f"{prefix}_coordinates_global_range_valid"] = (finite & global_range).astype(bool)
        result[f"{prefix}_zero_coordinate_flag"] = zero.fillna(False).astype(bool)
        result[f"{prefix}_near_zero_coordinate_flag"] = near_zero.fillna(False).astype(bool)
        result[f"{prefix}_outside_india_region_flag"] = (~india_region).fillna(True).astype(bool)

        status = pd.Series("valid", index=result.index, dtype="string")
        status.loc[~finite] = "missing_non_numeric_or_non_finite"
        status.loc[finite & ~global_range] = "outside_global_coordinate_range"
        status.loc[finite & global_range & zero] = "zero_coordinate"
        status.loc[finite & global_range & ~zero & ~india_region] = "outside_expected_india_region"
        result[f"{prefix}_coordinates_status"] = status

    result["coordinates_globally_valid"] = (
        result["restaurant_coordinates_global_range_valid"]
        & result["delivery_coordinates_global_range_valid"]
    )
    result["coordinates_in_expected_india_region"] = (
        ~result["restaurant_outside_india_region_flag"]
        & ~result["delivery_outside_india_region_flag"]
    )
    result["coordinates_valid"] = (
        result["coordinates_globally_valid"]
        & result["coordinates_in_expected_india_region"]
        & ~result["restaurant_zero_coordinate_flag"]
        & ~result["delivery_zero_coordinate_flag"]
    )
    result["invalid_coordinate_reason"] = result.apply(
        lambda row: ";".join(
            part for part in (
                f"restaurant:{row['restaurant_coordinates_status']}" if row["restaurant_coordinates_status"] != "valid" else "",
                f"delivery:{row['delivery_coordinates_status']}" if row["delivery_coordinates_status"] != "valid" else "",
            ) if part
        ) or "valid",
        axis=1,
    )
    return result


def calculate_delivery_distance(data: pd.DataFrame) -> pd.DataFrame:
    """Add Haversine straight-line km only for coordinate-valid records."""
    result = data.copy()
    if "coordinates_valid" not in result:
        result = validate_coordinates(result)
    valid = result["coordinates_valid"].fillna(False)
    distance = pd.Series(np.nan, index=result.index, dtype=float)
    if valid.any():
        lat1 = np.radians(pd.to_numeric(result.loc[valid, "restaurant_latitude"], errors="coerce"))
        lon1 = np.radians(pd.to_numeric(result.loc[valid, "restaurant_longitude"], errors="coerce"))
        lat2 = np.radians(pd.to_numeric(result.loc[valid, "delivery_location_latitude"], errors="coerce"))
        lon2 = np.radians(pd.to_numeric(result.loc[valid, "delivery_location_longitude"], errors="coerce"))
        delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        ).clip(0, 1)
        distance.loc[valid] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(haversine))
    result["approx_delivery_distance_km"] = distance
    return result


def detect_iqr_outliers(
    data: pd.DataFrame,
    columns: Iterable[str] = (
        "time_taken_min", "delivery_person_ratings", "delivery_person_age", "multiple_deliveries"
    ),
    multiplier: float = 1.5,
    example_limit: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Flag Tukey-IQR candidates and report fences/counts/examples; never delete.

    IQR flags are screening aids only. Sparse, skewed, or discrete variables
    can legitimately be flagged (for example, multiple_deliveries == 3).
    """
    if multiplier <= 0:
        raise ValueError("IQR multiplier must be positive.")
    result = data.copy()
    reports: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in result:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        finite = values.notna() & np.isfinite(values)
        observed = values.loc[finite]
        flag_column = f"{column}_iqr_outlier_flag"
        flag = pd.Series(pd.NA, index=result.index, dtype="boolean")
        if observed.empty:
            result[flag_column] = flag
            reports[column] = {
                "valid_count": 0, "q1": None, "q3": None,
                "lower_fence": None, "upper_fence": None,
                "outlier_count": 0, "outlier_percent": 0.0, "examples": [],
            }
            continue
        q1, q3 = observed.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = float(q1 - multiplier * iqr)
        upper_fence = float(q3 + multiplier * iqr)
        outlier_mask = finite & ((values < lower_fence) | (values > upper_fence))
        flag.loc[finite] = outlier_mask.loc[finite].astype(bool)
        result[flag_column] = flag
        example_columns = [c for c in ("id", column) if c in result]
        examples = result.loc[outlier_mask, example_columns].head(example_limit).copy()
        examples.insert(0, "source_row_index", examples.index)
        reports[column] = {
            "valid_count": int(finite.sum()),
            "q1": float(q1), "q3": float(q3),
            "lower_fence": lower_fence, "upper_fence": upper_fence,
            "outlier_count": int(outlier_mask.sum()),
            "outlier_percent": float(outlier_mask.sum() / max(int(finite.sum()), 1) * 100),
            "examples": examples.astype(object).where(pd.notna(examples), None).to_dict("records"),
        }
    return result, reports


def flag_tentative_rating_range(data: pd.DataFrame) -> pd.DataFrame:
    """Flag ratings outside tentative 1–5 range without modifying the rating."""
    result = data.copy()
    if "delivery_person_ratings" in result:
        rating = pd.to_numeric(result["delivery_person_ratings"], errors="coerce")
        flag = pd.Series(pd.NA, index=result.index, dtype="boolean")
        observed = rating.notna() & np.isfinite(rating)
        flag.loc[observed] = ((rating.loc[observed] < 1) | (rating.loc[observed] > 5)).astype(bool)
        result["delivery_person_ratings_outside_tentative_1_5_flag"] = flag
    return result


def detect_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Report and flag exact-row/ID duplicates; retain every record."""
    result = data.copy()
    exact = result.duplicated(keep=False)
    excess_exact = result.duplicated(keep="first")
    id_column = "id" if "id" in result else None
    duplicate_id = result[id_column].duplicated(keep=False) if id_column else pd.Series(False, index=result.index)
    result["exact_duplicate_record_flag"] = exact.astype(bool)
    result["duplicate_id_flag"] = duplicate_id.astype(bool)
    return result, {
        "exact_duplicate_excess_count": int(excess_exact.sum()),
        "exact_duplicate_participating_rows": int(exact.sum()),
        "duplicate_id_excess_count": int(result[id_column].duplicated().sum()) if id_column else 0,
        "duplicate_id_participating_rows": int(duplicate_id.sum()),
        "policy": "Flag only; do not drop records without a documented resolution rule.",
    }


def create_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/day and multiple-delivery indicator features if available."""
    result = data.copy()
    if "order_date_parsed" in result:
        result["order_day"] = result["order_date_parsed"].dt.day.astype("Int64")
        result["day_of_week"] = result["order_date_parsed"].dt.day_name().astype("string")
    if "multiple_deliveries" in result:
        multiple = pd.to_numeric(result["multiple_deliveries"], errors="coerce")
        # Preserve Week 1 behavior (>1); source semantics of value 1 are unclear.
        # Confirm whether values count combined orders or additional drops.
        result["has_multiple_deliveries"] = (multiple > 1).astype("boolean")
    if "vehicle_condition" in result:
        result["vehicle_condition"] = pd.to_numeric(result["vehicle_condition"], errors="coerce").astype("Int64")
    return result


def prepare_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Use pandas category dtype for known categoricals; keep labels readable."""
    result = data.copy()
    category_columns = [*CATEGORY_EXPECTATIONS, "day_of_week"]
    for column in category_columns:
        if column in result:
            result[column] = result[column].astype("string").fillna("Unknown").astype("category")
    return result


def fit_standard_scaler(
    training_data: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Fit optional standard-scaling statistics using training rows only.

    Scaling is not applied by the default cleaner. Tree models do not need it,
    and raw units aid interpretation. Use this helper only for a scale-sensitive
    model; never fit it on test/validation data or geographic lat/lon used with
    Haversine distance. Missing values remain missing.
    """
    parameters: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(training_data[column], errors="coerce")
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        parameters[column] = {"mean": mean, "scale": scale if scale > 0 else 1.0}
    return parameters


def apply_standard_scaler(
    data: pd.DataFrame,
    parameters: dict[str, dict[str, float]],
    suffix: str = "_standardized",
) -> pd.DataFrame:
    """Add optional standardized columns while preserving original values."""
    result = data.copy()
    for column, values in parameters.items():
        if column not in result:
            raise KeyError(f"Scaling column is missing: {column}")
        numeric = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}{suffix}"] = (numeric - values["mean"]) / values["scale"]
    return result


def flag_numeric_domain_values(data: pd.DataFrame) -> pd.DataFrame:
    """Flag nonpositive delivery durations without deleting or changing them."""
    result = data.copy()
    if "time_taken_min" in result:
        duration = pd.to_numeric(result["time_taken_min"], errors="coerce")
        observed = duration.notna() & np.isfinite(duration)
        flag = pd.Series(pd.NA, index=result.index, dtype="boolean")
        flag.loc[observed] = (duration.loc[observed] <= 0).astype(bool)
        result["time_taken_min_nonpositive_flag"] = flag
    return result


def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Run conservative preprocessing and return all original rows plus flags."""
    result = clean_column_names(data)
    result, _ = parse_order_date(result)
    result, _ = parse_order_and_pickup_times(result)
    result = handle_missing_values(result)
    result = validate_coordinates(result)
    result = calculate_delivery_distance(result)
    result = flag_numeric_domain_values(result)
    result = create_derived_features(result)
    result, _ = detect_iqr_outliers(result)
    result = flag_tentative_rating_range(result)
    result, _ = detect_duplicates(result)
    return prepare_categorical_variables(result)


def _missing_summary(data: pd.DataFrame) -> dict[str, int]:
    """Return nonzero per-column missing counts."""
    counts = data.isna().sum()
    return {str(column): int(count) for column, count in counts.items() if count}


def _numeric_summary(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Report observed ranges and non-numeric/non-finite values by numeric field."""
    output: dict[str, dict[str, Any]] = {}
    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column]) and column not in NUMERIC_SOURCE_COLUMNS:
            continue
        converted = pd.to_numeric(data[column], errors="coerce")
        source_present = data[column].notna()
        finite = converted.notna() & np.isfinite(converted)
        valid = converted.loc[finite]
        output[str(column)] = {
            "dtype": str(data[column].dtype),
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "non_numeric_or_nonfinite_count": int((source_present & ~finite).sum()),
        }
    return output


def _category_audit(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return observed values/counts and values outside documented expectations."""
    output: dict[str, dict[str, Any]] = {}
    for column, expected in CATEGORY_EXPECTATIONS.items():
        if column not in data:
            continue
        counts = data[column].value_counts(dropna=False)
        unexpected = [value for value in counts.index if pd.notna(value) and value not in expected and value != "Unknown"]
        output[column] = {
            "unique_count_including_missing": int(data[column].nunique(dropna=False)),
            "value_counts": {str(key): int(value) for key, value in counts.items()},
            "unexpected_values_preserved": [str(value) for value in unexpected],
        }
    return output


def build_audit_summary(source: pd.DataFrame, processed: pd.DataFrame) -> dict[str, Any]:
    """Build a reusable JSON-serializable before/after quality audit."""
    raw = clean_column_names(source)
    date_diag = _date_diagnostics(raw)
    time_diag = _time_diagnostics(raw)
    coordinate_counts = {
        "restaurant_status_counts": processed["restaurant_coordinates_status"].value_counts().to_dict(),
        "delivery_status_counts": processed["delivery_coordinates_status"].value_counts().to_dict(),
        "globally_valid_pair_rows": int(processed["coordinates_globally_valid"].sum()),
        "within_expected_india_region_rows": int(processed["coordinates_in_expected_india_region"].sum()),
        "zero_coordinate_rows": int((processed["restaurant_zero_coordinate_flag"] | processed["delivery_zero_coordinate_flag"]).sum()),
        "near_zero_coordinate_rows": int((processed["restaurant_near_zero_coordinate_flag"] | processed["delivery_near_zero_coordinate_flag"]).sum()),
        "valid_for_distance_rows": int(processed["coordinates_valid"].sum()),
        "distance_not_calculated_rows": int(processed["approx_delivery_distance_km"].isna().sum()),
    }
    _, outlier_reports = detect_iqr_outliers(raw)
    duplicate_summary = {
        "exact_duplicate_excess_count": int(raw.duplicated().sum()),
        "exact_duplicate_participating_rows": int(raw.duplicated(keep=False).sum()),
        "duplicate_id_excess_count": int(raw["id"].duplicated().sum()) if "id" in raw else 0,
        "duplicate_id_participating_rows": int(raw["id"].duplicated(keep=False).sum()) if "id" in raw else 0,
        "policy": "Flag only; do not drop records without a documented resolution rule.",
    }
    return {
        "source_documentation": {
            "local_file": "Zomato Dataset.csv",
            "candidate_public_listing": "https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset",
            "verification_status": "Candidate listing only; the exact identity of this local file and its original provenance have not been verified.",
            "license_collection_method_publication_date": "Not asserted; confirm from the original publisher before documenting.",
        },
        "source": {
            "rows": int(len(raw)), "columns": int(len(raw.columns)),
            "dtypes": {str(k): str(v) for k, v in raw.dtypes.items()},
            "missing_by_column": _missing_summary(raw),
            "numeric_ranges": _numeric_summary(raw),
            "categories": _category_audit(raw),
        },
        "processed": {
            "rows": int(len(processed)), "columns": int(len(processed.columns)),
            "dtypes": {str(k): str(v) for k, v in processed.dtypes.items()},
            "missing_by_column": _missing_summary(processed),
            "numeric_ranges": _numeric_summary(processed),
            "categories": _category_audit(processed),
        },
        "duplicates": duplicate_summary,
        "date_parsing": date_diag,
        "time_parsing": time_diag,
        "coordinates": coordinate_counts,
        "outliers": outlier_reports,
        "tentative_rating_range": {
            "range": [1, 5],
            "flagged_count": int(processed["delivery_person_ratings_outside_tentative_1_5_flag"].fillna(False).sum()),
            "note": "Tentative only; confirm the rating scale with authoritative dataset documentation.",
        },
        "category_normalization": {
            "known_label_case_or_whitespace_changes": _normalization_counts(raw),
            "policy": "Trim whitespace and case-normalize only labels matching the observed canonical vocabulary; preserve other labels.",
        },
        "field_descriptions": FIELD_DESCRIPTIONS,
        "scaling": "No scaling applied by default; optional scaler learns statistics from training data only.",
    }


def _date_diagnostics(data: pd.DataFrame) -> dict[str, Any]:
    raw = data.get("order_date", pd.Series(pd.NA, index=data.index))
    parsed = pd.to_datetime(raw, format="%d-%m-%Y", errors="coerce")
    missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
    failed = ~missing & parsed.isna()
    return {
        "format_assumption": "%d-%m-%Y based on observed source strings",
        "source_missing_count": int(missing.sum()),
        "parse_failure_count": int(failed.sum()),
        "parsed_missing_count": int(parsed.isna().sum()),
        "parsed_success_count": int(parsed.notna().sum()),
        "parsed_min": str(parsed.min().date()) if parsed.notna().any() else None,
        "parsed_max": str(parsed.max().date()) if parsed.notna().any() else None,
        "failure_examples": _failure_examples(data, "order_date", failed),
    }


def _time_diagnostics(data: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in ("time_orderd", "time_order_picked"):
        raw = data.get(column, pd.Series(pd.NA, index=data.index))
        parsed = raw.map(parse_time_to_minutes)
        missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
        failed = ~missing & parsed.isna()
        output[column] = {
            "source_missing_count": int(missing.sum()),
            "parse_failure_count": int(failed.sum()),
            "parsed_missing_count": int(parsed.isna().sum()),
            "parsed_success_count": int(parsed.notna().sum()),
            "failure_examples": _failure_examples(data, column, failed),
        }
    output["assumptions"] = {
        "decimal_strings": "Values from 0 inclusive to 1 exclusive are treated as fractions of a day.",
        "value_1": "Not interpreted as midnight; reported as a parse failure.",
        "24_hour_clock": "24:xx values are not normalized without verified source convention.",
        "overnight_interval": "Negative pickup-minus-order clock differences are rolled forward by one day as an assumption; intervals over 12 hours are set missing and flagged.",
    }
    return output


def _normalization_counts(data: pd.DataFrame) -> dict[str, int]:
    """Compute reproducible counts of category values that normalization changes."""
    _, counts = normalize_categories(data)
    return counts


def print_audit_summary(summary: dict[str, Any]) -> None:
    """Print the key audit details and representative diagnostic examples."""
    print("DATA QUALITY AUDIT — source versus Week 2 output")
    for label in ("source", "processed"):
        stage = summary[label]
        print(f"\n{label.title()}: {stage['rows']:,} rows x {stage['columns']} columns")
        print("Dtypes:", json.dumps(stage["dtypes"], ensure_ascii=False))
        print("Missing by column:", json.dumps(stage["missing_by_column"], ensure_ascii=False))
    print("\nDuplicates:", json.dumps(summary["duplicates"], ensure_ascii=False))
    print("\nDate parsing:", json.dumps(summary["date_parsing"], ensure_ascii=False))
    print("\nTime parsing:", json.dumps(summary["time_parsing"], ensure_ascii=False))
    print("\nCoordinate checks:", json.dumps(summary["coordinates"], ensure_ascii=False))
    print("\nUnexpected category values:")
    for column, details in summary["source"]["categories"].items():
        print(f"  {column}: {details['unexpected_values_preserved'] or 'none'}")
    print("\nNumeric observed ranges:")
    for column, details in summary["source"]["numeric_ranges"].items():
        print(f"  {column}: min={details['min']}, max={details['max']}, nonnumeric/nonfinite={details['non_numeric_or_nonfinite_count']}")
    print("\nIQR outlier screening (flags only; no values or rows deleted):")
    for column, details in summary["outliers"].items():
        print(
            f"  {column}: fences=({details['lower_fence']}, {details['upper_fence']}), "
            f"flagged={details['outlier_count']}/{details['valid_count']} "
            f"({details['outlier_percent']:.2f}%), examples={details['examples']}"
        )
    print("\nTentative rating range:", json.dumps(summary["tentative_rating_range"], ensure_ascii=False))
    print("\nCategory normalization:", json.dumps(summary["category_normalization"], ensure_ascii=False))
    print("\nScaling:", summary["scaling"])
    print("\nField descriptions are available in FIELD_DESCRIPTIONS; ambiguous definitions remain marked for confirmation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Original CSV input path.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH.name}; Week 1 cleaned file is not overwritten).",
    )
    parser.add_argument(
        "--audit-output", type=Path, default=DEFAULT_AUDIT_PATH,
        help=f"Machine-readable JSON audit path (default: {DEFAULT_AUDIT_PATH}).",
    )
    args = parser.parse_args()

    original = load_dataset(args.input)
    processed = preprocess_dataset(original)
    summary = build_audit_summary(original, processed)
    summary["paths"] = {
        "input": str(args.input.resolve()),
        "processed_csv": str(args.output.resolve()),
        "audit_json": str(args.audit_output.resolve()),
    }
    summary["row_retention"] = {
        "source_rows": int(len(original)),
        "output_rows": int(len(processed)),
        "rows_removed": int(len(original) - len(processed)),
        "policy": "No rows removed; malformed values, duplicates, coordinate concerns, and outliers are retained and flagged.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output, index=False)
    with args.audit_output.open("w", encoding="utf-8") as audit_file:
        json.dump(summary, audit_file, indent=2, ensure_ascii=False, allow_nan=False)
    print_audit_summary(summary)
    print(f"\nWeek 2 dataset written to: {args.output.resolve()}")
    print(f"Audit summary written to: {args.audit_output.resolve()}")


if __name__ == "__main__":
    main()
