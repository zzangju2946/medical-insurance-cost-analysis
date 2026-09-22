# Medical Insurance Cost Analysis & Prediction

A Python actuarial analytics portfolio project exploring how **BMI and smoking status are associated with medical charges**, with special attention to a BMI × smoking interaction and out-of-sample prediction.

> **Scope:** The dataset's target is `charges` (medical charges), **not an actual insurance premium**. Results are observational, based on a small public dataset, and should not be used to set insurance prices.

## Research question

Does the relationship between BMI and medical charges differ for smokers and non-smokers? Does adding the interaction improve held-out prediction performance?

## Dataset

- Source: [Medical Insurance Cost Dataset on Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)
- Original: 1,338 observations, 7 columns (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).
- Preparation: no missing observations; removed one exact duplicate row, leaving **1,337** rows. Because there is no customer ID, duplicate identity cannot be confirmed.
- Source data are **not bundled with this README**; download from Kaggle and review its usage terms.

## Approach

1. Explore medical charges against BMI using a scatterplot.
2. Fit OLS models: `charges ~ bmi`, `charges ~ bmi + C(smoker)`, and `charges ~ bmi * C(smoker)`.
3. Compare full-dataset exploratory R², adjusted R², and AIC.
4. Split the data 80/20 (1,069 train and 268 test), stratified by smoking status, with `random_state=42`.
5. Refit each model on training data only and evaluate test MAE, RMSE, and R².

## Main findings

**Full-data exploratory model fit (not test performance):**

| Model | R² | Adjusted R² | AIC |
|---|---:|---:|---:|
| BMI only | 0.0394 | 0.0386 | 28,884.01 |
| BMI + Smoking | 0.6579 | 0.6574 | 27,505.41 |
| BMI × Smoking | 0.7418 | 0.7412 | 27,131.23 |

The BMI × smoking coefficient was **approximately +$1,389.77 per BMI unit** (reported p < 0.001). The fitted BMI slope was **approximately $83.34 for non-smokers** and **$1,473.11 for smokers**. These reflect associations, not causal effects.

**Held-out test performance (268 records):**

| Model | MAE ($) | RMSE ($) | Test R² |
|---|---:|---:|---:|
| BMI only | 8,961.56 | 11,608.82 | 0.0651 |
| BMI + Smoking | 4,826.16 | 6,177.59 | 0.7353 |
| BMI × Smoking | 4,323.58 | 5,594.39 | 0.7829 |

In this single split, adding the interaction reduced test RMSE versus the additive model by about 9.4%. Predictions remain imperfect for individual high-cost observations.

## Figures

- `BMI vs Medical insurance charges.png`: exploratory BMI scatterplot.
- `bmi_smoking_interaction.png`: fitted lines by smoking status (full cleaned data).
- `actual_vs_predicted.png`: actual versus predicted charges (held-out test data).
- `medical_insurance_analysis_report.pdf`: illustrated analysis report.

## Local reproduction

The original analysis script is maintained in your local project folder as `insurance_analysis.py`. A typical local structure is:

```text
Actuarial Analysis/
├── insurance_analysis.py             # Add your original script
├── insurance.csv                     # Download separately from Kaggle
├── README.md
├── medical_insurance_analysis_report.pdf
├── BMI vs Medical insurance charges.png
├── bmi_smoking_interaction.png
└── actual_vs_predicted.png
```

Run from the project directory (after activating your virtual environment):

```bash
python -m pip install pandas numpy matplotlib statsmodels scikit-learn
python insurance_analysis.py
```

The script may also generate `model_comparison.csv`, `model_evaluation.csv`, and text summaries.

## Limitations

The dataset is limited in size, predictor information, and documented provenance; it has no policy exposure or insurance premium field. The two-predictor model does not control for other characteristics. OLS assumptions were not comprehensively checked, and results are from a single train/test split. Replicated validation, residual diagnostics, and a multivariable or positive-cost GLM would be natural next steps.
