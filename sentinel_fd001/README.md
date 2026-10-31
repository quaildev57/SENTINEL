# SENTINEL: FD001 baseline

End-to-end baseline on NASA C-MAPSS **FD001** (single operating condition, single fault mode).

```
pip install pandas scikit-learn matplotlib mlflow sqlalchemy alembic
python sentinel_fd001.py --data <folder with train_FD001.txt, test_FD001.txt, RUL_FD001.txt> --out results
mlflow ui --backend-store-uri sqlite:///results/mlflow.db      # view tracked runs
```
Runtime is about 45 seconds. Seeds are fixed.

## Pipeline
1. **Data engineering:** drops 7 flat sensors (keeps 14), median-imputes gaps, standardises (fit on training engines only), builds 10-cycle rolling mean, std and slope features.
2. **Anomaly detection:** Isolation Forest trained on the first 40 cycles of each engine (healthy baseline). Score is smoothed over 5 cycles; the alert threshold is the 99th percentile of healthy-baseline scores. An engine is escalated after 3 consecutive flags.
3. **RUL prediction:** average of HistGradientBoosting and RandomForest, target capped at 125 cycles (standard piecewise-linear RUL).
4. **Triage agent (rule-based, no LLM):** Critical if predicted RUL <= 30; High if <= 60 or 3+ consecutive anomaly flags; Watch if <= 100; otherwise Healthy. Cases go to a human if the two models disagree by more than 25 cycles, or if the anomaly flags are sustained while predicted RUL looks healthy.
5. **Work orders:** drafts with severity, predicted RUL, top-3 deviating sensors as evidence and a recommended action. Status is always "DRAFT - awaiting approval".
6. **MLOps:** every run is logged in MLflow. Retraining demo: train on 60 engines, a new batch of 20 arrives, a PSI drift check runs, the model is retrained and compared.

## Results (see `results/metrics.json`)
| Item | Value |
|---|---|
| Test RMSE (100 official test engines, last cycle) | 15.4 cycles (HistGB 15.0, RF 16.2) |
| NASA score | 381 |
| Anomaly lead time (20 held-out training engines) | median 53.5 cycles before failure, range 26 to 144, 0 engines never flagged |
| False-flag rate while RUL > 150 | 0.0% |
| Triage vs true severity (test fleet) | exact 72%, within one level 100% |
| True-critical engines caught as Critical or High | 25 of 25 |
| Retraining demo | max PSI 0.21; validation RMSE 13.4 (v1) vs 13.6 (v2): retraining did not improve it |

## Caveats
- FD001 only, single seed, 100 test engines. FD002 to FD004 (6 operating regimes, 2 fault modes) are not done.
- Triage thresholds (30 / 60 / 100) were set by judgement and checked on the test fleet. They were **not** tuned on a separate validation set yet.
- The "needs human review" rule fired for 0 of 100 test engines, so it is untested in practice.
- Anomaly lead time and false-flag rate come from 20 engines.
