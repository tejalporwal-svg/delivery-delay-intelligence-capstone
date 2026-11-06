# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from src.shipment_prep import clean_shipments, build_features

def run():
    # Load new shipments & metadata
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # Load model and configurations
    model = joblib.load("models/delay_model.pkl")
    threshold_val = float(pd.read_csv("models/delay_threshold.csv").iloc[0, 0])
    features_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))
    
    # Load segment information
    segments_df = pd.read_csv("outputs/hub_segments.csv")
    
    # 1. Clean shipments using medians
    clean_df, _ = clean_shipments(new_raw, medians=medians)
    
    # 2. Build features
    feat_df = build_features(clean_df, hubs)
    
    # Re-order columns
    X_new = feat_df[features_order]
    
    # 3. Predict probabilities
    probs = model.predict_proba(X_new)[:, 1]
    
    # Round probabilities to 4 decimal places
    feat_df["delay_probability"] = np.round(probs, 4)
    feat_df["flagged"] = (feat_df["delay_probability"] >= threshold_val).astype(int)
    
    # 4. Join Hub segment
    # segment contains hub_id and segment
    seg_map = dict(zip(segments_df["hub_id"], segments_df["segment"]))
    feat_df["segment"] = feat_df["hub_id"].map(seg_map)
    
    # 5. Determine operational action
    # Escalate to hub manager when flagged and Hub's segment is Escalate
    # Proactive customer call when flagged and other segment
    # No action when not flagged
    def get_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            else:
                return "Proactive customer call"
        else:
            return "No action"
            
    feat_df["action"] = feat_df.apply(get_action, axis=1)
    
    # Save actionable results in order of file
    output_df = feat_df[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    output_df.to_csv("outputs/new_shipment_actions.csv", index=False)

if __name__ == "__main__":
    run()
