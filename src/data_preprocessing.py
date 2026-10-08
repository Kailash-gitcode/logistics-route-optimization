"""Beginner-friendly preprocessing for the Zomato delivery dataset.

Run from the project root with: python src/data_preprocessing.py

The source CSV is only read. The script writes its cleaned result to
data/cleaned_logistics_data.csv.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "Zomato Dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_logistics_data.csv"

# The observations describe deliveries in India. These broad bounds catch
# this dataset's zero/negative placeholder-like coordinates and impossible
# values while allowing locations across India. Invalid values are flagged
# and replaced with NaN before distance is calculated; they are never treated
# as actual locations.
INDIA_LATITUDE_BOUNDS = (6.0, 38.0)
INDIA_LONGITUDE_BOUNDS = (68.0, 98.0)
EARTH_RADIUS_KM = 6371.0


def load_dataset(csv_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load the CSV while keeping identifier and mixed-format time fields intact."""
    return pd.read_csv(csv_path, low_memory=False)


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with readable, consistent snake_case column names."""
    cleaned = data.copy()
    cleaned.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()))
        .strip("_")
        .lower()
        for name in cleaned.columns
    ]
    return cleaned


def parse_order_date(data: pd.DataFrame) -> pd.DataFrame:
    """Parse day-month-year dates; invalid or missing dates become NaT."""
    result = data.copy()
    if "order_date" in result:
        # The source uses values such as 12-02-2022 (day-month-year).
        result["order_date"] = pd.to_datetime(
            result["order_date"], format="%d-%m-%Y", errors="coerce"
        )
    return result


def parse_time_to_minutes(value: object) -> float:
    """Convert a clock time or Excel-style fraction of a day to minutes.

    Decimal-looking values are accepted only when they are numeric fractions
    from 0 (inclusive) to 1 (exclusive), which unambiguously represents a
    fraction of a 24-hour day. Other malformed values safely return NaN.
    """
    if pd.isna(value):
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan

    # Excel can store a time as the fraction of a day (e.g. 0.5 = noon).
    try:
        numeric_value = float(text)
        if 0 <= numeric_value < 1:
            return numeric_value * 24 * 60
        return math.nan
    except (TypeError, ValueError):
        pass

    # Accept HH:MM or HH:MM:SS, and reject out-of-range clock components.
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if not match:
        return math.nan
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    if hour > 23 or minute > 59 or second > 59:
        return math.nan
    return hour * 60 + minute + second / 60


def parse_order_and_pickup_times(data: pd.DataFrame) -> pd.DataFrame:
    """Add parsed minute-of-day fields and a midnight-aware order-to-pickup interval."""
    result = data.copy()
    for source, target in (
        ("time_orderd", "order_time_minutes"),
        ("time_order_picked", "pickup_time_minutes"),
    ):
        if source in result:
            result[target] = result[source].map(parse_time_to_minutes)
        else:
            result[target] = math.nan

    # A negative interval is assumed to cross midnight and is rolled forward.
    # Intervals over 12 hours are treated as unreliable rather than retained.
    interval = result["pickup_time_minutes"] - result["order_time_minutes"]
    interval = interval.where(interval >= 0, interval + 24 * 60)
    result["order_to_pickup_minutes"] = interval.where(interval <= 12 * 60)
    result["order_hour"] = (result["order_time_minutes"] // 60).astype("Int64")
    result["pickup_hour"] = (result["pickup_time_minutes"] // 60).astype("Int64")
    return result


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Keep rows; mark absent categories as Unknown and preserve numeric NaNs.

    No blanket row deletion or arbitrary numeric imputation is performed.
    Missing values in measurements such as age, ratings, time and coordinates
    remain missing so downstream analysis can choose a suitable strategy.
    """
    result = data.copy()
    categorical_columns = [
        "delivery_person_id", "weather_conditions", "road_traffic_density",
        "type_of_order", "type_of_vehicle", "festival", "city",
    ]
    for column in categorical_columns:
        if column in result:
            result[column] = result[column].astype("string").str.strip()
            result[column] = result[column].fillna("Unknown").replace("", "Unknown")
    return result


def validate_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Flag invalid coordinate pairs and set their coordinate values to NaN.

    This dataset represents Indian deliveries, so zero, non-finite values, and
    coordinates outside broad India bounds are treated as invalid. Both the
    restaurant and destination must be valid for the row to have a distance.
    """
    result = data.copy()
    coordinate_columns = [
        "restaurant_latitude", "restaurant_longitude",
        "delivery_location_latitude", "delivery_location_longitude",
    ]
    for column in coordinate_columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        else:
            result[column] = math.nan

    rest_ok = (
        result["restaurant_latitude"].between(*INDIA_LATITUDE_BOUNDS)
        & result["restaurant_longitude"].between(*INDIA_LONGITUDE_BOUNDS)
    )
    delivery_ok = (
        result["delivery_location_latitude"].between(*INDIA_LATITUDE_BOUNDS)
        & result["delivery_location_longitude"].between(*INDIA_LONGITUDE_BOUNDS)
    )
    result["restaurant_coordinates_valid"] = rest_ok.fillna(False)
    result["delivery_coordinates_valid"] = delivery_ok.fillna(False)
    result["coordinates_valid"] = result["restaurant_coordinates_valid"] & result[
        "delivery_coordinates_valid"
    ]
    result["invalid_coordinate_reason"] = ""
    result.loc[~result["restaurant_coordinates_valid"], "invalid_coordinate_reason"] += (
        "invalid_restaurant;"
    )
    result.loc[~result["delivery_coordinates_valid"], "invalid_coordinate_reason"] += (
        "invalid_delivery;"
    )

    # Replace each invalid pair with NaN so it cannot accidentally be used as a location.
    for valid_column, latitude, longitude in (
        ("restaurant_coordinates_valid", "restaurant_latitude", "restaurant_longitude"),
        ("delivery_coordinates_valid", "delivery_location_latitude", "delivery_location_longitude"),
    ):
        invalid = ~result[valid_column]
        result.loc[invalid, [latitude, longitude]] = math.nan
    return result


def calculate_delivery_distance(data: pd.DataFrame) -> pd.DataFrame:
    """Add approximate great-circle distance in km for validated coordinate pairs."""
    result = data.copy()
    distance = pd.Series(math.nan, index=result.index, dtype="float64")
    valid = result.get("coordinates_valid", pd.Series(False, index=result.index)).fillna(False)
    if valid.any():
        lat1 = result.loc[valid, "restaurant_latitude"].map(math.radians)
        lon1 = result.loc[valid, "restaurant_longitude"].map(math.radians)
        lat2 = result.loc[valid, "delivery_location_latitude"].map(math.radians)
        lon2 = result.loc[valid, "delivery_location_longitude"].map(math.radians)
        delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
        haversine = (
            (delta_lat / 2).map(math.sin) ** 2
            + lat1.map(math.cos) * lat2.map(math.cos) * (delta_lon / 2).map(math.sin) ** 2
        ).clip(0, 1)
        distance.loc[valid] = 2 * EARTH_RADIUS_KM * haversine.map(
            lambda value: math.asin(math.sqrt(value))
        )
    result["approx_delivery_distance_km"] = distance
    return result


def create_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, delivery, and simple operational features where available."""
    result = data.copy()
    if "order_date" in result:
        result["order_day"] = result["order_date"].dt.day.astype("Int64")
        result["day_of_week"] = result["order_date"].dt.day_name().astype("string")
    if "multiple_deliveries" in result:
        multiple = pd.to_numeric(result["multiple_deliveries"], errors="coerce")
        result["has_multiple_deliveries"] = (multiple > 1).astype("boolean")
    if "vehicle_condition" in result:
        result["vehicle_condition"] = pd.to_numeric(
            result["vehicle_condition"], errors="coerce"
        ).astype("Int64")
    return result


def prepare_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical text and mark categorical columns for later models.

    Category dtype is useful in pandas analysis; CSV output stores the labels
    as readable text. Encoding should be fit on training data during modeling.
    """
    result = data.copy()
    category_columns = [
        "weather_conditions", "road_traffic_density", "type_of_order",
        "type_of_vehicle", "festival", "city", "day_of_week",
    ]
    for column in category_columns:
        if column in result:
            result[column] = result[column].astype("string").fillna("Unknown").astype("category")
    return result


def preprocess_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Run the reusable preprocessing steps and return all original rows."""
    result = clean_column_names(data)
    result = parse_order_date(result)
    result = parse_order_and_pickup_times(result)
    result = handle_missing_values(result)
    result = validate_coordinates(result)
    result = calculate_delivery_distance(result)
    result = create_derived_features(result)
    return prepare_categorical_variables(result)


def print_preprocessing_summary(
    original: pd.DataFrame, cleaned: pd.DataFrame
) -> None:
    """Print a compact audit summary for the internship project."""
    original_missing = int(original.isna().sum().sum())
    cleaned_missing = int(cleaned.isna().sum().sum())
    invalid_coordinates = int((~cleaned["coordinates_valid"]).sum())
    valid_distances = int(cleaned["approx_delivery_distance_km"].notna().sum())
    dates = cleaned["order_date"].dropna() if "order_date" in cleaned else pd.Series(dtype="datetime64[ns]")
    date_range = f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "Unavailable"
    print("Preprocessing summary")
    print(f"Original rows/columns: {original.shape[0]} / {original.shape[1]}")
    print(f"Cleaned rows/columns:  {cleaned.shape[0]} / {cleaned.shape[1]}")
    print(f"Missing values (before/after): {original_missing} / {cleaned_missing}")
    print(f"Rows with invalid coordinates: {invalid_coordinates}")
    print(f"Valid distance calculations: {valid_distances}")
    print(f"Parsed date range: {date_range}")
    print(
        "Derived features: order_day, day_of_week, order_hour, pickup_hour, "
        "approx_delivery_distance_km, has_multiple_deliveries, vehicle_condition, "
        "road_traffic_density, weather_conditions"
    )


def main() -> None:
    """Load, preprocess, summarize, and save the cleaned dataset."""
    original = load_dataset(DEFAULT_INPUT_PATH)
    cleaned = preprocess_dataset(original)
    cleaned.to_csv(DEFAULT_OUTPUT_PATH, index=False)
    print_preprocessing_summary(original, cleaned)
    print(f"Cleaned dataset saved to: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
