# ea-game-financial-viability

Machine learning framework to predict the financial viability of Steam Early Access (EA) games using pre-launch and early-lifecycle signals.

AI in Finance project, B.Tech CSFT, Manipal Institute of Technology, MAHE.
Team: J Sheshanth (240958242), Krishi Reddem (240958009), Anirudh Syam Menon (240958039)

## Idea

Take games tagged **Early Access in an old Steam snapshot** (May 2019) and look at a **newer snapshot** (Nov 2026) to see what happened to them:

| Status | Meaning | n |
|---|---|---|
| `graduated` | no longer tagged Early Access (full release) | 888 |
| `still_ea` | still tagged Early Access (stalled) | 1,705 |
| `removed` | no longer on the store (abandoned / delisted) | 361 |

All features come from the **old** snapshot and all labels from the **new** one, so there is no time leakage.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM (classification); Cox proportional hazards, Kaplan-Meier and cause-specific Cox (graduation vs removal); SHAP for explainability; out-of-fold risk tiers as a "financial risk curve".

## Data

Raw datasets are not stored in this repo (size / licence). Download them from Kaggle:

| Role | Dataset | Link |
|---|---|---|
| OLD snapshot (May 2019) | Steam Store Games (Clean dataset), Nik Davis, file `steam.csv` | https://www.kaggle.com/datasets/nikdavis/steam-store-games |
| NEW snapshot (Nov 2026) | Steam Games Dataset, Fronkon Games, file `games.csv` | https://www.kaggle.com/datasets/fronkongames/steam-games-dataset |

**Processed dataset used for modelling** (EA games with labels and engineered features): [`ea_outputs/ea_labeled_dataset.csv`](ea_outputs/ea_labeled_dataset.csv)

## Run

```bash
pip install -r requirements.txt
python inspect_data.py steam.csv games.csv          # check headers first
python ea_pipeline_v2.py --old steam.csv --new games.csv --old-date 2019-05-01 --new-date 2026-11-02
# python ea_pipeline_v2.py --demo --repeats 1       # synthetic sanity check only (numbers are NOT results)
```

Outputs are written to `ea_outputs/`; the full console log is in `run_log.txt`.

## Results (real data)

**Classification** (3x repeated 5-fold CV; baseline graduated rate = 34.2%, n = 2,593)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.752 | 0.617 | 0.608 |
| Random Forest | 0.746 | 0.601 | 0.588 |
| XGBoost | 0.739 | 0.598 | 0.580 |
| LightGBM | 0.737 | 0.597 | 0.582 |

Sensitivity (removed games counted as failures, n = 2,954, baseline 30.1%): ROC-AUC 0.740 / 0.743 / 0.738 / 0.735 (LogReg / RF / XGB / LightGBM).
Single 80/20 hold-out ROC-AUC is about 0.69-0.70; repeated CV is the main result.

**Survival analysis**: Cox concordance 0.659. Cause-specific Cox: graduation 0.664, removal 0.583.
Higher positive-review ratio speeds graduation (HR 1.41) and lowers removal hazard (HR 0.87).

**Risk tiers** (out-of-fold score; realised outcome %):

| Tier | n | graduated | removed | still EA |
|---|---|---|---|---|
| Low | 985 | 52.6 | 11.0 | 36.4 |
| Medium | 984 | 25.2 | 13.2 | 61.6 |
| High | 985 | 12.4 | 12.5 | 75.1 |

**Label validation** (median OLD to NEW change): review volume growth 1.099 (graduated) vs 0.234 (still EA), playtime change 0.377 vs 0.000 (both p < 0.001); review-score change -0.038 vs -0.025 (p = 0.18, not significant).

Plots: `ea_outputs/shap_summary.png`, `km_curves.png`, `cox_coefficients.png`, `cox_cause_specific.png`, `risk_tiers.png`, `risk_curves_by_tier.png`.

## Limitations and scope adjustments

- Proposal features not available in the public snapshots (wishlists, patch frequency, concurrent-player trends) are not used; reviews and playtime act as early-lifecycle proxies.
- Exit dates are unknown, so graduation and removal times are imputed at the midpoint between the two snapshots; timing in KM curves is approximate.
- The proportional-hazards assumption is violated for `log_owners`, `log_price` and `reviews_per_month`; hazard ratios are average effects.
- Labels are proxies for viability, not revenue. Only games still in EA at the 2019 snapshot are included (survivorship).
- Removal is only weakly predictable from these features (concordance 0.583).
- SHAP and Cox effects are associations, not causal effects.
