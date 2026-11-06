# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run():
    # 1. Hub Profile
    df = pd.read_csv("data/modelling_table.csv")
    
    # Group by hub_id
    hub_prof = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).reset_index().sort_values("hub_id")
    
    # Round columns of the profile table
    hub_prof_rounded = hub_prof.copy()
    for c in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        hub_prof_rounded[c] = hub_prof_rounded[c].round(4)
    hub_prof_rounded.to_csv("outputs/hub_profile.csv", index=False)
    
    # 2. K-Means segmentation
    profile_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hub_prof[profile_cols])
    
    # Fit K=3
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    km3.fit(X_scaled)
    labels = km3.labels_
    
    # 3. Separation evidence for K = 2, 3, 4, 5
    def compute_across_over_within(X, labels_k):
        n = len(X)
        within_dists = []
        across_dists = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                if labels_k[i] == labels_k[j]:
                    within_dists.append(dist)
                else:
                    across_dists.append(dist)
        mean_w = np.mean(within_dists) if len(within_dists) > 0 else 1e-9
        mean_a = np.mean(across_dists) if len(across_dists) > 0 else 1e-9
        return round(mean_a / mean_w, 4)

    sep_records = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        ratio = compute_across_over_within(X_scaled, km.labels_)
        # count sizes of clusters
        sizes = pd.Series(km.labels_).value_counts().sort_values(ascending=False).astype(str).tolist()
        sep_records.append({
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": "-".join(sizes)
        })
    pd.DataFrame(sep_records).to_csv("outputs/part_d_separation.csv", index=False)
    
    # 4. Segment names and tier check
    # delay_rate mean per cluster label to map correctly to Standard, Watch, Escalate
    hub_prof["cluster_label"] = labels
    cl_means = hub_prof.groupby("cluster_label")["delay_rate"].mean().sort_values()
    # order is Standard (lowest), Watch (middle), Escalate (highest)
    map_dict = {
        cl_means.index[0]: "Standard",
        cl_means.index[1]: "Watch",
        cl_means.index[2]: "Escalate"
    }
    
    hub_prof["segment"] = hub_prof["cluster_label"].map(map_dict)
    
    # Read hubs.csv to get cities, regions, tiers
    hubs_info = pd.read_csv("data/hubs.csv")
    hub_segments_full = hub_prof.merge(hubs_info, on="hub_id", how="left")
    
    # outputs/hub_segments.csv
    hub_segments_full[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)
    
    # outputs/segment_profile.csv
    # profile means are means of the hub-level values
    seg_profile = hub_segments_full.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    ).reindex(["Standard", "Watch", "Escalate"]).reset_index()
    
    for c in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        seg_profile[c] = seg_profile[c].round(4)
    seg_profile.to_csv("outputs/segment_profile.csv", index=False)
    
    # outputs/tier_vs_segment_mismatch.csv
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments_full["expected_segment"] = hub_segments_full["hub_tier"].map(tier_map)
    mismatches = hub_segments_full[hub_segments_full["segment"] != hub_segments_full["expected_segment"]]
    mismatches[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

if __name__ == "__main__":
    run()
