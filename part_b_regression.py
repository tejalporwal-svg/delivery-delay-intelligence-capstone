# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run():
    # Load modelling table
    df = pd.read_csv("data/modelling_table.csv")
    
    from src.shipment_prep import FEATURES
    
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 1. Train-validation-test splits (840 / 180 / 180)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    # The remaining 85% is train_val (1020). Split it to get 840 train, 180 validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=180/1020, random_state=42
    )
    
    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())
    
    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 12)),
        ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4))
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)
    
    # Helper helper to calculate metrics
    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return round(mae, 4), round(mse, 4), round(rmse, 4), round(r2, 4)
    
    # 3. Baseline (Mean of training y)
    mean_y_train = y_train.mean()
    baseline_preds = np.full(shape=y_val.shape, fill_value=mean_y_train)
    b_mae, b_mse, b_rmse, b_r2 = get_metrics(y_val, baseline_preds)
    
    # 4. Tree depth diagnostics
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_records = []
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
        va_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        depth_records.append({
            "max_depth": "unlimited" if d is None else str(d),
            "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(va_rmse, 4)
        })
    pd.DataFrame(depth_records).to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # Determine best tree depth by lowest validation RMSE
    best_depth_idx = np.argmin([r["val_rmse"] for r in depth_records])
    best_depth_val = depths[best_depth_idx]
    
    # 5. Four candidates on validation split
    # Candidate A: Linear Regression in a Pipeline
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    
    # Candidate B: Decision Tree from Task 4
    dt_best = DecisionTreeRegressor(max_depth=best_depth_val, random_state=42)
    dt_best.fit(X_train, y_train)
    dt_preds = dt_best.predict(X_val)
    
    # Candidate C: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    
    # Candidate D: Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_val)
    
    val_metrics = [
        ("mean_baseline", "val", b_mae, b_mse, b_rmse, b_r2),
        ("linear_regression", "val", *get_metrics(y_val, lr_preds)),
        ("decision_tree", "val", *get_metrics(y_val, dt_preds)),
        ("random_forest", "val", *get_metrics(y_val, rf_preds)),
        ("gradient_boosting", "val", *get_metrics(y_val, gb_preds))
    ]
    
    # 6. Learning rate experiment
    lrs = [0.01, 0.1, 1.0]
    lr_records = []
    for lr_val in lrs:
        model_lr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr_val, random_state=42)
        model_lr.fit(X_train, y_train)
        preds_lr = model_lr.predict(X_val)
        _, _, rmse_lr, r2_lr = get_metrics(y_val, preds_lr)
        lr_records.append({
            "learning_rate": lr_val,
            "val_rmse": rmse_lr,
            "val_r2": r2_lr
        })
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 7. Cross-validated comparison and final model
    # Find the top 2 validation models (based on val RMSE in val_metrics)
    # The candidates sorted by val RMSE
    candidate_results = val_metrics[1:] # Exclude mean_baseline
    candidate_results.sort(key=lambda x: x[4]) # index 4 is RMSE
    best_cands = [candidate_results[0][0], candidate_results[1][0]]
    
    # Build standard definitions
    models_dict = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth_val, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []
    for cand_name in best_cands:
        model_cv = models_dict[cand_name]
        scores = cross_val_score(model_cv, X_train_val, y_train_val, cv=kf)
        cv_records.append({
            "model": cand_name,
            "mean_r2": round(scores.mean(), 4),
            "std_r2": round(scores.std(), 4)
        })
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)
    
    # Final model selection based on lowest validation RMSE
    final_model_name = candidate_results[0][0]
    final_model = models_dict[final_model_name]
    # Train final model on train, evaluate once on test
    final_model.fit(X_train, y_train)
    test_preds = final_model.predict(X_test)
    test_mae, test_mse, test_rmse, test_r2 = get_metrics(y_test, test_preds)
    
    val_metrics.append((final_model_name, "test", test_mae, test_mse, test_rmse, test_r2))
    
    pd.DataFrame(val_metrics, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]).to_csv(
        "outputs/part_b_metrics.csv", index=False
    )

if __name__ == "__main__":
    run()
