# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def run():
    # Load modelling table
    df = pd.read_csv("data/modelling_table.csv")
    
    from src.shipment_prep import FEATURES, NOISE_COLUMNS
    
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 1. Stratified split (840 / 180 / 180)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=180/1020, random_state=42, stratify=y_train_val
    )
    
    # Metrics collector helper
    def eval_clf(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return int(tn), int(fp), int(fn), int(tp), round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)
    
    # 2. Baseline & Candidates
    # Blind baseline: all on time (0 delay predictions)
    blind_preds = np.zeros_like(y_val)
    b_tn, b_fp, b_fn, b_tp, b_acc, b_prec, b_rec, b_f1 = eval_clf(y_val, blind_preds)
    
    # Logistic Regression Pipeline
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    lr_metrics = eval_clf(y_val, lr_preds)
    
    # Decision Tree: select depth from 2,3,4,5,6,8,10
    best_dt_d = 2
    best_dt_f1 = -1.0
    best_dt_metrics = None
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_val)
        m = eval_clf(y_val, dt_preds)
        if m[7] > best_dt_f1:  # F1 index is 7
            best_dt_f1 = m[7]
            best_dt_d = d
            best_dt_metrics = m
        elif m[7] == best_dt_f1:
            if d < best_dt_d:
                best_dt_d = d
                best_dt_metrics = m
                
    # Random Forest Default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_def_preds = rf_def.predict(X_val)
    rf_def_metrics = eval_clf(y_val, rf_def_preds)
    
    models_records = [
        ("blind_all_on_time", b_tn, b_fp, b_fn, b_tp, b_acc, b_prec, b_rec, b_f1),
        ("logistic_regression", *lr_metrics),
        (f"decision_tree_depth_{best_dt_d}", *best_dt_metrics),
        ("random_forest_default", *rf_def_metrics)
    ]
    pd.DataFrame(models_records, columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).to_csv(
        "outputs/part_c_models.csv", index=False
    )
    
    # 3. Feature importances and ANOVA scores
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    importances = rf_all.feature_importances_
    
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_
    
    # Rank ANOVA f_scores
    # higher score -> rank 1 (descending order)
    f_rank_series = pd.Series(f_scores).rank(ascending=False, method="min").astype(int)
    
    feat_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": [round(x, 4) for x in f_scores],
        "f_rank": f_rank_series.values,
        "rf_importance": [round(x, 4) for x in importances],
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feat_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    # SelectKBest with different values of k
    k_vals = [5, 7, 10, 12, 17]
    k_records = []
    for kv in k_vals:
        pipe_k = Pipeline([
            ("selector", SelectKBest(f_classif, k=kv)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_k.fit(X_train, y_train)
        k_preds = pipe_k.predict(X_val)
        k_f1 = f1_score(y_val, k_preds, zero_division=0)
        k_records.append({
            "k": kv,
            "val_f1": round(k_f1, 4)
        })
    pd.DataFrame(k_records).to_csv("outputs/part_c_selectk.csv", index=False)
    
    # 4. Tuning with fixed k = 7
    selector_7 = SelectKBest(f_classif, k=7)
    X_train_7 = selector_7.fit_transform(X_train, y_train)
    X_val_7 = selector_7.transform(X_val)
    
    # Parameter Grid
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }
    
    # Grid Search
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5
    )
    gs.fit(X_train_7, y_train)
    gs_best_params = gs.best_params_
    gs_best_cv_acc = gs.best_score_
    gs_val_preds = gs.predict(X_val_7)
    gs_val_f1 = f1_score(y_val, gs_val_preds, zero_division=0)
    
    # Randomized Search
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42
    )
    rs.fit(X_train_7, y_train)
    rs_best_params = rs.best_params_
    rs_best_cv_acc = rs.best_score_
    rs_val_preds = rs.predict(X_val_7)
    rs_val_f1 = f1_score(y_val, rs_val_preds, zero_division=0)
    
    # Untuned Forest on 7 features
    untuned_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    untuned_7.fit(X_train_7, y_train)
    untuned_val_preds = untuned_7.predict(X_val_7)
    untuned_val_f1 = f1_score(y_val, untuned_val_preds, zero_division=0)
    
    tuning_records = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(gs_best_params),
            "best_cv_accuracy": round(gs_best_cv_acc, 4),
            "val_f1": round(gs_val_f1, 4)
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rs_best_params),
            "best_cv_accuracy": round(rs_best_cv_acc, 4),
            "val_f1": round(rs_val_f1, 4)
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": round(untuned_val_f1, 4)
        }
    ]
    pd.DataFrame(tuning_records).to_csv("outputs/part_c_tuning.csv", index=False)
    
    # 5. Choose the final configuration
    # Pick best based on highest val_f1
    # sorting logic: untuned_7 -> grid_search -> randomized_search on tie
    # Let's write standard selector:
    t_df = pd.DataFrame(tuning_records)
    # custom sort order: default_forest_same_7_features (0), grid_search (1), randomized_search (2)
    sort_mapping = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2
    }
    t_df["sort_order"] = t_df["search"].map(sort_mapping)
    t_df = t_df.sort_values(by=["val_f1", "sort_order"], ascending=[False, True])
    winner_row = t_df.iloc[0]
    winner_name = winner_row["search"]
    
    if winner_name == "grid_search":
        final_params = gs_best_params
    elif winner_name == "randomized_search":
        final_params = rs_best_params
    else:
        final_params = {}
        
    final_clf = RandomForestClassifier(random_state=42, **final_params)
    final_pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", final_clf)
    ])
    final_pipe.fit(X_train, y_train)
    
    # 6. Alert threshold analysis
    val_probs = final_pipe.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    thresh_records = []
    chosen_t = 0.1
    for t in thresholds:
        t_preds = (val_probs >= t).astype(int)
        t_tn, t_fp, t_fn, t_tp, t_acc, t_prec, t_rec, t_f1 = eval_clf(y_val, t_preds)
        thresh_records.append({
            "threshold": t,
            "TN": t_tn, "FP": t_fp, "FN": t_fn, "TP": t_tp,
            "accuracy": t_acc, "precision": t_prec, "recall": t_rec, "f1": t_f1
        })
        
    thresh_df = pd.DataFrame(thresh_records)
    thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # Operational Rule: highest threshold where validation recall is at least 0.80. Else 0.1
    met_recall = thresh_df[thresh_df["recall"] >= 0.80]
    if len(met_recall) > 0:
        chosen_t = float(met_recall["threshold"].max())
    else:
        chosen_t = 0.1
        
    # 7. One test evaluation
    test_probs = final_pipe.predict_proba(X_test)[:, 1]
    test_final_preds = (test_probs >= chosen_t).astype(int)
    tst_tn, tst_fp, tst_fn, tst_tp, tst_acc, tst_prec, tst_rec, tst_f1 = eval_clf(y_test, test_final_preds)
    
    final_test_df = pd.DataFrame([{
        "threshold": chosen_t,
        "TN": tst_tn, "FP": tst_fp, "FN": tst_fn, "TP": tst_tp,
        "accuracy": tst_acc, "precision": tst_prec, "recall": tst_rec, "f1": tst_f1
    }])
    final_test_df.to_csv("outputs/part_c_final_test.csv", index=False)
    
    # Final configuration save
    final_config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": final_params.get("n_estimators", 100),
        "max_depth": final_params.get("max_depth", "unlimited"),
        "min_samples_leaf": final_params.get("min_samples_leaf", 1),
        "threshold": chosen_t,
        "chosen_from": winner_name
    }])
    final_config_df.to_csv("outputs/part_c_final_config.csv", index=False)

if __name__ == "__main__":
    run()
