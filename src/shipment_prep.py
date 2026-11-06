# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    # 1. Drop exact duplicates
    clean_df = raw.drop_duplicates().copy()
    
    # 2. service_type: title case & strip
    clean_df["service_type"] = clean_df["service_type"].astype(str).str.strip().str.title()
    
    # 3. payment_mode: replace missing with "Unknown"
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
    
    # 4. declared_value: clean rupee & comma, make float
    clean_df["declared_value_inr"] = clean_df["declared_value"].astype(str)\
        .str.replace("\u20b9", "", regex=True)\
        .str.replace(",", "", regex=True)\
        .replace("nan", np.nan)\
        .astype(float)
    clean_df = clean_df.drop(columns=["declared_value"])
    
    # 5. dispatch_time to datetime
    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
    
    # 6. weight sentinel
    clean_df.loc[clean_df["parcel_weight_kg"] > 100.0, "parcel_weight_kg"] = np.nan
    
    # 7. Fill missing with median
    if medians is None:
        medians = {
            "distance_km": float(clean_df["distance_km"].median()),
            "parcel_weight_kg": float(clean_df["parcel_weight_kg"].median()),
            "declared_value_inr": float(clean_df["declared_value_inr"].median())
        }
    
    clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    return clean_df, medians

def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df
