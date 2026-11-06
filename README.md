# Delhivery Last-Mile Delay Intelligence Pipeline

This repository contains the complete production-grade pipeline solving last-mile delay estimation, proactively flagging at-risk shipments, and aligning operational playbooks to actual hub performance tierings.

## How to Run

1. **Setup Environment**:
   Clone this repository on your local system, then navigate to its root and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependencies are locked with exact versions including `pandas==2.2.3`, `scikit-learn==1.5.2`, `numpy`, and `joblib`)*.

2. **Run Pipeline Steps**:
   Execute the following scripts in sequence:
   ```bash
   # Part A: Data Foundation & Cleaning
   python part_a_clean_eda.py

   # Part B: Delivery-Hours Regression Models
   python part_b_regression.py

   # Part C: Delay Classification & Operational Thresholds
   python part_c_classification.py

   # Part D: Hub Segment Profiling & Clustering
   python part_d_segments.py

   # Package model and configurations
   python package_model.py

   # Batch Predict for New Shipments
   python predict_delay.py
   ```
   *Note: `models/delay_model.pkl` is fully regenerated when running `package_model.py`.*

---

## Data Findings

- **Service Type Delays**: **Express** shipments are delayed most frequently with a delay rate of **0.4897** (48.97%), compared to **0.2056** for Standard and **0.0857** for Economy.
- **Hub Tier Impact**: The actual delay rate increases from Tier 1 (**0.0400**) to Tier 2 (**0.2150**) and escalates sharply in Tier 3 (**0.4400**).
- **Distance Correlations**: Shorter distances under 200 km show a delay rate of just **0.0219**. This increases to **0.1368** (201-400 km), **0.4364** (401-600 km), and reaches **0.8523** for long distances over 600 km (601+ band).
- **Strongest Correlated Columns**: The columns most strongly correlated with actual `delivery_hours` are:
  1. `promised_hours` (Correlation: **0.8373**)
  2. `distance_km` (Correlation: **0.4013**)
  3. `n_stops` (Correlation: **0.3121**)
- **Log Transformation Justification**: Before log-transformation, `declared_value_inr` exhibits extreme skewness with a mean of **INR 1,533.05** and a median of **INR 1,121.50** (strongly right-skewed). After natural log-transformation (`log_declared_value`), the mean is **7.0280** and the median is **7.0220**, producing a highly symmetrical, normal-like distribution that stabilizes variance for distance-based estimators and linear models.

---

## Model Results (Regression)

- **Baseline Mean Model**: Baseline $R^2$ on the validation split is **-0.0210**, which indicates that predicting the historical mean is worse than a simple horizontal line and explains none of the variance.
- **Validation Winners**: Both **Linear Regression** ($R^2$ = **0.9312**, RMSE = **6.1664**) and **Gradient Boosting** ($R^2$ = **0.9308**, RMSE = **6.1863**) performed exceptionally well. In the 5-fold cross-validation, Gradient Boosting won with a mean $R^2$ of **0.9354** (std: **0.0094**) compared to Linear Regression's mean $R^2$ of **0.9255** (std: **0.0101**).
- **Final Test Performance**: Gradient Boosting evaluated on the held-out test partition produced an $R^2$ of **0.9490** (RMSE = **5.2758** hours).
- **Tree Depth Diagnostic**: Unlimited depth tree ($max\_depth=None$) exhibits severe high-variance overfitting (Train RMSE = **0.0000** but Validation RMSE = **7.7946**). $max\_depth=1$ represents high-bias underfitting (Train RMSE = **11.4554**, Validation RMSE = **12.1328**). The optimal balanced tree depth was identified at **d=6** (Validation RMSE = **6.2818**).
- **Learning Rates**: A learning rate of **0.1** is optimal ($R^2$ = **0.9308**). At **0.01**, the model learns too slowly underfitting within 100 trees ($R^2$ = **0.7130**), while at **1.0**, the gradient descent overshoots local minima, degrading performance ($R^2$ = **0.8440**).

---

## Model Results (Classification)

- **Blind Baseline**: Accuracy is **0.7667** by simply predicting no delays, which is operationally useless as it misses 100% of actual delays (0% recall).
- **Feature Importances & Noise**: The 5 bookkeeping noise columns (`label_print_batch`, `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`) hold a combined feature importance of **0.2248** (22.48%) in the raw random forest. However, in ANOVA F-score ranking, all 5 noise features successfully rank outside the top 7 (ranking 8th or higher with F-scores below 10, whereas our top 7 signal features remain above 30).
- **Hyperparameter Tuning**: Hyperparameter tuning on the validation set did not surpass the validation F1 of the untuned forest on the same 7 features (**0.8333**). On a small 180-row validation set, validation-driven tuning often overfits to local validation noise rather than learning generalizable signals.
- **Calibrated Threshold**: Using our operations rule to maximize catch-rate (recall $\ge 80\%$), the calibrated threshold is **0.5**. This calibrates a precision of **0.8000** and recall of **0.8571** on the validation set.
- **Test Set Performance**: On the independent test split, the model achieved a test precision of **0.6889**, a test recall of **0.7561**, and a test F1 of **0.7209** at the 0.5 threshold.

---

## Operations Recommendation

1. **Segment Profiles & Playbooks**:
   - **Standard Segment** (7 Hubs, Delay Rate **0.0257**): Operations should execute standard protocols.
   - **Watch Segment** (8 Hubs, Delay Rate **0.1550**): Implement buffer hours at dispatch.
   - **Escalate Segment** (9 Hubs, Delay Rate **0.4600**): Must execute intensive playbooks. The hubs in this segment are: **HUB07, HUB13, HUB15, HUB17, HUB18, HUB19, HUB20, HUB21, HUB23**.

2. **Tier-vs-Segment Mismatches**:
   - **HUB07 Kolkata** (Tier 1 -> Watch)
   - **HUB13 Nagpur** (Tier 2 -> Escalate)
   - **HUB15 Kochi** (Tier 2 -> Escalate)
   - **HUB22 Dehradun** (Tier 3 -> Watch)
   *Operations should override literal labels in resource planning; Nagpur and Kochi must receive the Escalate Playbook.*

3. **Calibrated Operations Rule**: At our **0.5** alert threshold, out of every 10 flagged parcels, approximately **6.9** are genuinely delayed (68.9% precision). Out of every 10 actual delayed parcels, the model successfully catches **7.6** of them (75.6% recall).

4. **Batch Prediction Summary**: For the **48** new shipments, our pipeline successfully flagged **8** shipments. **4** of these flagged shipments originate from **Escalate** hubs, automatically generating **Escalate to hub manager** tasks, while the remaining **4** trigger **Proactive customer calls**.

5. **Segment Stability**: We maintained $K=3$ because lower K does not provide sufficient granularity, and higher $K$ (4 or 5) generates clusters with fewer than 6 hubs, which are too small to support their own playbooks.

---

## Production Plan & Limitations

- **Production Readiness mapping**:
  - *Consistency*: Shared functions in `src/shipment_prep.py` are strictly reused in training (`part_a_clean_eda.py`) and scoring (`predict_delay.py`).
  - *Reproducibility*: Locked seeds (`random_state=42`) and pinned versions in `requirements.txt`.
  - *Maintainability*: Modular design separating ETL, training, clustering, and inferencing.
- **Monitoring Metrics & Triggers**:
  - *Weekly delayed share*: Trigger alert if actual weekly delays drift $\pm 10\%$ from the baseline average of **23.17%**.
  - *Weekly alert rate*: Escalation trigger if total flagged shipments exceed **20%** of weekly volumes.
  - *Realized Recall*: Operational alert if realized recall drops below **65%**.
- **Data Drift & Sale Windows**: Models should be retrained monthly. The 15–28 September festival sale window (`sale_event_flag`) experiences an average **12-hour surge** in delays. The model must only compare sale periods with past sale-period distributions, never ordinary weeks.
- **Limitations**: Synthetic dataset assumptions, small test sample size (41 actual delays), and single-split validation bias for threshold tuning.
