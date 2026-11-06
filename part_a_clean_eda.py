# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from src.shipment_prep import clean_shipments, build_features, PROMISED_HOURS

def run():
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # Collect metrics before filling to write in checks
    dup_removed = len(raw) - len(raw.drop_duplicates())
    
    # Let's clean it
    clean_df, med_dict = clean_shipments(raw, medians=None)
    
    # service type spellings before & after
    raw_dedup = raw.drop_duplicates().copy()
    spellings_before = raw_dedup["service_type"].nunique()
    spellings_after = clean_df["service_type"].nunique()
    
    # Missing counts before median filling
    # Distance, weight, value, payment mode
    # (Weight sentinel needs to be filtered out of raw before counting missing weight_before)
    missing_distance_km_before = raw_dedup["distance_km"].isna().sum()
    
    raw_weights_no_sentinel = raw_dedup["parcel_weight_kg"].copy()
    sentinel_weight_rows = (raw_weights_no_sentinel > 100.0).sum()
    raw_weights_no_sentinel.loc[raw_weights_no_sentinel > 100.0] = np.nan
    missing_parcel_weight_before = raw_weights_no_sentinel.isna().sum()
    
    # Value converter
    def clean_val_str(v):
        if pd.isna(v): return np.nan
        return float(str(v).replace("\u20b9", "").replace(",", ""))
    raw_vals = raw_dedup["declared_value"].map(clean_val_str)
    missing_declared_value_before = raw_vals.isna().sum()
    
    missing_payment_mode_before = raw_dedup["payment_mode"].isna().sum()
    
    # Build features
    modelling_table = build_features(clean_df, hubs)
    modelling_table["is_delayed"] = (modelling_table["delivery_hours"] > modelling_table["promised_hours"]).astype(int)
    
    # Write modelling table
    modelling_table.to_csv("data/modelling_table.csv", index=False)
    
    # Compute checks
    # Hours outliers and quantiles
    q1 = modelling_table["delivery_hours"].quantile(0.25)
    q3 = modelling_table["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((modelling_table["delivery_hours"] < lower_bound) | (modelling_table["delivery_hours"] > upper_bound)).sum()
    
    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", dup_removed),
        ("rows_after_dedup", len(clean_df)),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", spellings_before),
        ("service_type_spellings_after", spellings_after),
        ("declared_value_mean_inr", round(modelling_table["declared_value_inr"].mean(), 4)),
        ("declared_value_median_inr", round(modelling_table["declared_value_inr"].median(), 4)),
        ("log_declared_value_mean", round(modelling_table["log_declared_value"].mean(), 4)),
        ("log_declared_value_median", round(modelling_table["log_declared_value"].median(), 4)),
        ("median_distance_km", round(med_dict["distance_km"], 4)),
        ("median_parcel_weight_kg", round(med_dict["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(med_dict["declared_value_inr"], 4)),
        ("rows_after_hub_merge", len(modelling_table)),
        ("nulls_in_features_and_target", int(modelling_table.isnull().sum().sum())),
        ("delayed_count", int(modelling_table["is_delayed"].sum())),
        ("delay_rate", round(modelling_table["is_delayed"].mean(), 4)),
        ("hours_q1 / hours_q3", f"{round(q1, 2)} / {round(q3, 2)}"),
        ("hours_lower_bound / hours_upper_bound", f"{round(lower_bound, 2)} / {round(upper_bound, 2)}"),
        ("hours_outlier_count", int(outliers))
    ]
    
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)
    
    # Delay by group
    # Dimension: service_type
    g_service = modelling_table.groupby("service_type").agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index().rename(columns={"service_type": "group"})
    g_service.insert(0, "dimension", "service_type")
    
    # Dimension: hub_tier
    g_tier = modelling_table.groupby("hub_tier").agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index().rename(columns={"hub_tier": "group"})
    g_tier["group"] = g_tier["group"].astype(str)
    g_tier.insert(0, "dimension", "hub_tier")
    
    # Dimension: distance_band
    # bins 0-200, 201-400, 401-600, 601+ (edges 0, 200, 400, 600, 2000)
    modelling_table["distance_band"] = pd.cut(
        modelling_table["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True
    )
    g_dist = modelling_table.groupby("distance_band", observed=False).agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index().rename(columns={"distance_band": "group"})
    g_dist["group"] = g_dist["group"].astype(str)
    g_dist.insert(0, "dimension", "distance_band")
    
    # Combine all delay by group
    delay_by_group = pd.concat([g_service, g_tier, g_dist], ignore_index=True)
    delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
    delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)
    delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # Correlations
    numeric_cols = modelling_table.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude trivial features if not required
    corr_matrix = modelling_table[numeric_cols].corr().round(4)
    # labels kept as first column
    corr_matrix.insert(0, "feature", corr_matrix.index)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=False)

if __name__ == "__main__":
    run()
