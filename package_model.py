# -*- coding: utf-8 -*-
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def run():
    # Read model configuration
    config_df = pd.read_csv("outputs/part_c_final_config.csv")
    winner_name = config_df.loc[0, "chosen_from"]
    threshold = float(config_df.loc[0, "threshold"])
    
    # Re-read training configuration parameter values if any
    n_est = int(config_df.loc[0, "n_estimators"])
    m_dep = config_df.loc[0, "max_depth"]
    min_l = int(config_df.loc[0, "min_samples_leaf"])
    
    max_depth_val = None if m_dep == "unlimited" else int(m_dep)
    
    # Rebuild stratified splits exactly
    df = pd.read_csv("data/modelling_table.csv")
    from src.shipment_prep import FEATURES
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=180/1020, random_state=42, stratify=y_train_val
    )
    
    # Instantiate model and build training pipelines
    final_clf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_depth_val,
        min_samples_leaf=min_l,
        random_state=42
    )
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", final_clf)
    ])
    pipeline.fit(X_train, y_train)
    
    # Save artefacts to models/
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/delay_model.pkl")
    
    # delay_threshold.csv
    pd.DataFrame([[threshold]], columns=["threshold"]).to_csv("models/delay_threshold.csv", index=False)
    
    # feature_order.csv
    pd.DataFrame(FEATURES, columns=["feature"]).to_csv("models/feature_order.csv", index=False)
    
    # impute_values.csv
    # Recompute or reload from modeling
    raw_df = pd.read_csv("data/raw_shipments.csv")
    from src.shipment_prep import clean_shipments
    _, medians = clean_shipments(raw_df, medians=None)
    medians_records = [{"column": k, "median": v} for k, v in medians.items()]
    pd.DataFrame(medians_records).to_csv("models/impute_values.csv", index=False)
    
    # Reload check
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    
    in_mem_probs = pipeline.predict_proba(X_test)[:, 1]
    reload_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]
    max_diff = float(np.abs(in_mem_probs - reload_probs).max())
    
    reload_checks = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", max_diff),
        ("threshold", threshold)
    ]
    pd.DataFrame(reload_checks, columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)

if __name__ == "__main__":
    run()
