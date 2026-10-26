# Predicting Employee Productivity Using Tree Models

Garment manufacturing is labour intensive, and hitting a production target depends heavily on how a team actually performs on the floor. This project builds decision tree and random forest regressors to predict `actual_productivity` for garment factory teams, then digs into which operational factors drive that number.

**Dataset:** 1,197 daily records from a garment factory ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees))

## Results

| Model | RMSE | MAE | R squared |
| --- | --- | --- | --- |
| **Tuned random forest** | **0.1148** | **0.0775** | **0.5033** |
| Tuned decision tree | 0.1335 | 0.0844 | 0.3287 |
| Unpruned decision tree | 0.1509 | 0.0862 | 0.1419 |
| Baseline (predict the mean) | 0.1635 | 0.1274 | -0.0064 |

The tuned random forest explains roughly half the variance in team productivity, a clear improvement over both a single tree and the mean baseline.

## What drives productivity

Random forest feature importances:

| Feature | Importance |
| --- | --- |
| `targeted_productivity` | 0.176 |
| `incentive` | 0.133 |
| `smv` (task complexity) | 0.105 |
| `team` | 0.097 |
| `no_of_workers` | 0.097 |
| `over_time` | 0.080 |

1. **Targets encode prior knowledge.** `targeted_productivity` dominates because targets are set per team based on what that team can realistically achieve.
2. **Incentive is the actionable lever.** It ranks second and is one of the few variables management directly controls.
3. **Complexity and team size hurt.** Higher `smv` and larger `no_of_workers` push productivity down, pointing at coordination overhead.

## Approach

- **Data cleaning** — the raw `department` column contains `sweing` (a typo) plus `finishing` with a trailing space, splitting one department across two labels. `wip` is missing for every finishing record, which is structural rather than lost data, so it is filled with 0 rather than dropping rows.
- **Feature engineering** — date decomposed into month and day-of-month, categoricals one-hot encoded.
- **Baseline first** — every model is measured against predicting the training mean.
- **Tuning** — `GridSearchCV` with 5-fold CV over depth and leaf constraints. The unpruned tree showed the classic failure mode (near-perfect training fit, weak test performance), which tuning corrected.
- **Diagnostics** — actual-vs-predicted and residual plots.

## Caveats

Mean-decrease-in-impurity importance is biased toward high-cardinality continuous features, so the ranking is indicative rather than causal. The residual plot also shows the model struggling at the low-productivity extreme, which is exactly where a factory would most want accuracy. A natural next step is reframing as classification: *will this team miss its target?*

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn

---

Dataquest guided project, completed as part of the Data Scientist in Python path.

Marco Anthony Ayuste