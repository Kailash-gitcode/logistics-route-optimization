# Food Delivery Logistics Route Optimization and Delivery Time Analysis

## 1. Project Overview

This Week 1 internship project explores how data science can support food-delivery logistics planning. It audits historical delivery records, summarizes delivery performance, predicts delivery duration, identifies exploratory geographic patterns, and demonstrates a small batch-workload optimization.

The analysis is descriptive and educational. It does not establish causal effects, define official service areas, or recommend real-world dispatch routes.

## 2. Business/Logistics Problem

Food-delivery operations need to understand how delivery duration varies with traffic, weather, vehicle, location, and order conditions. Better estimates and geographic summaries can inform service planning, staffing questions, and future dispatch research.

The available dataset provides historical order-level records, but not road-network travel times, dispatch events, courier availability, or promised delivery windows. The project therefore establishes analytical baselines and demonstrations rather than a deployable logistics system.

## 3. Objectives

- Inspect and clean the supplied delivery dataset without changing the original file.
- Explore delivery performance and define measurable operational KPIs.
- Compare a Linear Regression baseline with a Random Forest regression model for delivery-time prediction.
- Explore delivery-location density with geographic DBSCAN clustering.
- Illustrate how balancing delivery workload across batches can be expressed as a small optimization problem.
- Document data limitations and operational information needed for future work.

## 4. Dataset

The supplied file is `data/Zomato Dataset.csv`.

- **Records:** 45,584
- **Original columns:** 20
- **Date coverage:** February 11 to April 6, 2022 (44 distinct dates)
- **Target used for regression:** `Time_taken (min)`; the preprocessing script normalizes its name to `time_taken_min`.

The dataset includes order and delivery-person identifiers, driver age and rating, restaurant and delivery coordinates, order and pickup times, weather, traffic, vehicle details, multiple-delivery status, festival and city labels, and delivery duration.

## 5. Data Preprocessing

`src/data_preprocessing.py` reads the original CSV and, by default, writes the separate Week 2 output `data/week2_cleaned_logistics_data.csv` and `reports/week2_data_quality_summary.json`. It preserves the Week 1 cleaned file, retains rows with missing values, fills missing categorical labels with `Unknown`, and leaves numeric missing values available for later handling.

Coordinate pairs are validated before distance calculation. Zero, placeholder-like, and out-of-region coordinates are flagged and excluded from distance calculations. The preprocessing run retained all 45,584 rows and produced 41,513 valid approximate distance calculations; 4,071 rows were flagged for invalid coordinates.

`approx_delivery_distance_km` is a great-circle, straight-line approximation between restaurant and delivery coordinates. It is **not road distance, route length, or actual travel time**.

### Week 2 data quality audit

The Week 2 audit is documented in [`notebooks/05_week2_data_quality_audit.ipynb`](notebooks/05_week2_data_quality_audit.ipynb), with source limitations recorded in [`data/DATASET_SOURCE.md`](data/DATASET_SOURCE.md). The preprocessing retains source rows and raw values, preserves missing numeric measurements, labels unknown categories consistently, and flags parse failures, duplicates, coordinate concerns, and IQR outliers rather than silently repairing or deleting them. India-region bounds, the 1-5 rating range, and overnight pickup-time handling are assumptions that remain to be verified. The candidate Kaggle listing is not verified provenance for the local file.

## 6. Exploratory Data Analysis

`notebooks/01_data_audit_and_eda.ipynb` reviews row and column counts, data types, missingness, duplicates, category coverage, descriptive statistics, and delivery-time patterns by traffic, weather, vehicle, vehicle condition, multiple-delivery status, city, festival, weekday, and order hour.

The notebook generates figures in `visualizations/`, including delivery-time and distance distributions, group comparisons, and delivery time versus approximate distance. Distance charts and summaries exclude invalid coordinate pairs. Group differences are descriptive associations, not causal conclusions.

## 7. Key KPIs

The following are **observed or derived dataset metrics**, not contractual service commitments:

| KPI | Result | Definition |
|---|---:|---|
| Average delivery time | 26.29 min | Mean of recorded `time_taken_min` |
| Median delivery time | 26 min | Median of recorded `time_taken_min` |
| Long-delivery threshold | 32 min | 75th percentile of recorded duration |
| Long-delivery rate | 24.71% | Share strictly above the 75th-percentile threshold |
| Average approximate delivery distance | 9.72 km | Mean straight-line distance among 41,513 valid coordinate records |

The dataset has no promised delivery time. The 75th-percentile threshold is an **analytical proxy for the long-duration tail**, not an on-time delivery measure. A real operation should track on-time performance against the actual customer promise, plus cancellation/failure rate, cost per completed delivery, and courier utilization when the required data is available.

## 8. Regression Analysis

`src/modeling.py` and `notebooks/02_delivery_time_regression.ipynb` predict `time_taken_min` using available pre-delivery or assignment-time features, including approximate distance, traffic, weather, vehicle type and condition, multiple deliveries, city, festival, order hour, weekday, and driver age. IDs, driver ratings, pickup-related fields, and the target are excluded to reduce leakage and avoid using information unavailable at the intended prediction point.

The evaluation uses earlier dates for training and later dates for testing. In the recorded run, the cutoff was March 29, 2022, with 36,138 training and 9,446 test records. This chronological split better reflects prediction on future orders than a purely random split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 5.171 min | 6.501 min | 0.531 |
| Random Forest | **4.334 min** | **5.481 min** | **0.666** |

The Random Forest performed better on this holdout. MAE is the average absolute error in minutes; RMSE penalizes larger errors more; R² compares prediction error with a constant-mean baseline. These values are historical test results, not production guarantees.

**Environment note:** scikit-learn was unavailable during the recorded run, so the script used its documented NumPy fallback implementations. When scikit-learn is installed, the script uses its Linear Regression and Random Forest Regressor estimators. The fallback run is an educational implementation, not a tuned production model. Feature importance indicates which inputs the fitted forest used for prediction; it does not show causal impact.

## 9. Geographic Clustering

`src/clustering.py` and `notebooks/03_delivery_clustering.ipynb` use DBSCAN based primarily on validated delivery-location coordinates. The recorded configuration uses **eps = 1.0 km** and **min_samples = 20**. Haversine distance is used when scikit-learn is available; otherwise a SciPy unit-sphere search uses the mathematically equivalent chord radius.

- Valid geographic records: **41,513**
- Exploratory clusters: **280**
- Noise/outlier records: **7,708 (18.57%)**

Cluster summaries include record counts, average and median delivery duration, and average approximate delivery distance. These are exploratory geographic patterns, **not official dispatch zones**. Cluster count and noise share depend on the DBSCAN parameters. Real service-area design requires road-network data, actual travel times, demand by time, driver availability, fleet capacity, and operational constraints.

## 10. Optimization Demonstration

`src/optimization_demo.py` and `notebooks/04_optimization_demo.ipynb` demonstrate exact enumeration for a deliberately small batch-assignment problem. The example selects 12 real deliveries from a densely populated rounded destination-coordinate tile, assigns them to three hypothetical batches, and limits each batch to four deliveries.

The baseline assigns deliveries sequentially in groups of four. The illustrative optimized assignment minimizes the maximum batch workload (makespan), using historical `time_taken_min` values as hypothetical workload estimates:

| Measure | Baseline | Optimized |
|---|---:|---:|
| Deliveries | 12 | 12 |
| Batches/vehicles assumed | 3 | 3 |
| Capacity per batch | 4 | 4 |
| Maximum batch workload (makespan proxy) | 116 min | 104 min |

The makespan proxy improves by **12 minutes (10.3%)**. Total individual historical workload remains **310 minutes** in both assignments; the demonstration balances work across parallel batches rather than reducing delivery durations. Durations are historical outcomes with hindsight, not prospective dispatch estimates. This is an **illustrative assumption**, not a production vehicle-routing solution or a real-world route recommendation.

## 11. Key Findings

- Recorded delivery duration averages 26.29 minutes, with a median of 26 minutes.
- Delivery-time group means differ across traffic and weather labels. For example, mean duration was 31.18 minutes for Jam traffic and 21.27 minutes for Low traffic. These comparisons do not prove that traffic caused the difference.
- Multiple-delivery records averaged 41.59 minutes, compared with 25.46 minutes for records not marked as multiple deliveries. Batch sequence and composition are unavailable, so this is not an evaluation of a batching policy.
- Invalid coordinate pairs affect 4,071 records (8.9%); excluding them is important for spatial analysis.
- The Random Forest had lower MAE and RMSE and higher R² than Linear Regression on the chronological holdout.
- The batch-assignment demonstration lowered the maximum hypothetical batch workload from 116 to 104 minutes under its stated assumptions.

These findings are descriptive or illustrative; they should not be interpreted as causal effects.

## 12. Limitations

- The data covers a short period in 2022 and may not represent current demand, seasonality, or operating practices.
- Coordinate-derived distance is straight-line approximation, not road-network distance or route length.
- There are no promised delivery timestamps, so contractual on-time performance cannot be calculated.
- Road-network travel times, route sequences, order release times, actual travel-time matrices, driver availability, fleet size and capacity, shifts, costs, cancellation outcomes, and dispatch constraints are absent.
- Category groups can be very small, and missing or inconsistent records can affect comparisons.
- Regression metrics and feature importance do not establish causality or production readiness.
- DBSCAN clusters depend on parameter choices and are not official operating territories.
- The optimization uses realized historical duration as a workload proxy and balances batches without modeling roads, route interactions, or actual vehicle constraints.

## 13. Project Structure

```text
logistics-route-optimization/
├── data/
│   ├── Zomato Dataset.csv
│   ├── cleaned_logistics_data.csv
│   ├── week2_cleaned_logistics_data.csv
│   └── DATASET_SOURCE.md
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_delivery_time_regression.ipynb
│   ├── 03_delivery_clustering.ipynb
│   ├── 04_optimization_demo.ipynb
│   └── 05_week2_data_quality_audit.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── modeling.py
│   ├── clustering.py
│   └── optimization_demo.py
├── reports/
│   └── week2_data_quality_summary.json
├── visualizations/
└── README.md
```

## 14. How to Run the Project

Run commands from the project root. A Python environment with pandas, NumPy, SciPy, Pillow, and Jupyter is suitable. Scikit-learn is optional but recommended for the standard regression estimators and DBSCAN implementation.

```bash
python -m pip install pandas numpy scipy pillow scikit-learn jupyter
```

To recreate the cleaned dataset and figures:

```bash
python src/data_preprocessing.py
python src/modeling.py
python src/clustering.py
python src/optimization_demo.py
```

To open the analysis notebooks in Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in order. The modeling and clustering scripts use scikit-learn when available; the recorded run used the documented local fallbacks because package installation was unavailable in that environment.

## 15. Technologies Used

- Python
- pandas and NumPy for data handling and numerical calculations
- Pillow for saved charts in the lightweight execution environment
- SciPy `cKDTree` for the Haversine-equivalent DBSCAN fallback
- scikit-learn when available for Linear Regression, Random Forest, and Haversine DBSCAN
- Jupyter notebooks for the documented analysis
- Exact enumeration for the small illustrative batch assignment

## 16. Future Improvements

- Collect promised and actual event timestamps, including order release, ready, pickup, and arrival times.
- Add road-network routes, route distances, and time-dependent travel-time matrices.
- Record order demand by zone and time, driver availability, fleet size, vehicle capacity, and shift constraints.
- Add delivery windows, cancellation/failure outcomes, and operating costs.
- Validate coordinate quality and reassess clustering parameters with operational stakeholders.
- Evaluate predictive models on later time periods and compare against operational baselines before considering deployment.
- Extend the educational assignment model to include route sequencing, driver assignment, batching, delivery windows, fleet constraints, and route-cost minimization only after the necessary real-world data is available.
- Add a sourced background/literature review and compile the final internship report once the project analysis is complete.
