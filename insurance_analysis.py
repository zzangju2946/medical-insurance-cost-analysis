"""Medical insurance charge analysis and BMI x smoking prediction.

Download insurance.csv from the Kaggle source linked in README.md and place
it next to this script before running.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Use this script's folder so it works regardless of the Terminal location.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "insurance.csv"

# 1. Load and inspect the dataset.
df = pd.read_csv(DATA_FILE)
print("--- First Five Rows ---")
print(df.head())
print("\nOriginal dataset shape:", df.shape)
print("\n--- Data Information ---")
df.info()
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Duplicate Records ---")
print(df[df.duplicated(keep=False)])

# 2. Remove the one exact duplicate for this analysis.
# Without a customer ID, identical rows are not proof of the same customer.
df = df.drop_duplicates().copy()
print("\n--- Cleaned Dataset ---")
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

# 3. Exploratory scatterplot: BMI vs. medical charges.
plt.figure(figsize=(8, 5))
plt.scatter(df["bmi"], df["charges"], alpha=0.5)
plt.xlabel("BMI")
plt.ylabel("Medical Charges ($)")
plt.title("BMI vs. Medical Insurance Charges")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "BMI vs Medical insurance charges.png", dpi=300)
plt.close()  # Save without blocking subsequent calculations.

# 4. Fit full-data OLS models for exploratory coefficient interpretation.
# These are NOT the held-out evaluation models below.
model_bmi = smf.ols("charges ~ bmi", data=df).fit()
model_additive = smf.ols("charges ~ bmi + C(smoker)", data=df).fit()
model_interaction = smf.ols("charges ~ bmi * C(smoker)", data=df).fit()

print("\n--- BMI Regression Results ---")
print(model_bmi.summary())
(PROJECT_DIR / "bmi_ols_results.txt").write_text(
    model_bmi.summary().as_text(), encoding="utf-8"
)

print("\n--- BMI x Smoking Interaction Model ---")
print(model_interaction.summary())
(PROJECT_DIR / "bmi_smoking_interaction_results.txt").write_text(
    model_interaction.summary().as_text(), encoding="utf-8"
)

# 5. Display the interaction with separate smoker/non-smoker fitted lines.
bmi_range = np.linspace(df["bmi"].min(), df["bmi"].max(), 100)
non_smoker = pd.DataFrame({"bmi": bmi_range, "smoker": "no"})
smoker = pd.DataFrame({"bmi": bmi_range, "smoker": "yes"})

plt.figure(figsize=(9, 6))
for status, color in [("no", "blue"), ("yes", "red")]:
    subset = df[df["smoker"] == status]
    plt.scatter(
        subset["bmi"], subset["charges"],
        alpha=0.3, color=color, label=f"Smoker: {status}"
    )

plt.plot(
    bmi_range, model_interaction.predict(non_smoker),
    color="blue", linewidth=2, label="Non-smoker regression"
)
plt.plot(
    bmi_range, model_interaction.predict(smoker),
    color="red", linewidth=2, label="Smoker regression"
)
plt.xlabel("BMI")
plt.ylabel("Medical Charges ($)")
plt.title("BMI and Smoking Interaction on Medical Charges")
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_DIR / "bmi_smoking_interaction.png", dpi=300)
plt.close()

# 6. Compare full-data model fit (exploratory/in-sample only).
comparison = pd.DataFrame({
    "Model": ["BMI Only", "BMI + Smoking", "BMI x Smoking"],
    "R-squared": [model_bmi.rsquared, model_additive.rsquared, model_interaction.rsquared],
    "Adjusted R-squared": [
        model_bmi.rsquared_adj, model_additive.rsquared_adj, model_interaction.rsquared_adj
    ],
    "AIC": [model_bmi.aic, model_additive.aic, model_interaction.aic],
})
print("\n--- Model Comparison (Full Dataset) ---")
print(comparison.round(4).to_string(index=False))
comparison.to_csv(PROJECT_DIR / "model_comparison.csv", index=False)

# 7. Split before refitting models for held-out predictive evaluation.
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["smoker"]
)
print("\n--- Dataset Split ---")
print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)

formulas = {
    "BMI Only": "charges ~ bmi",
    "BMI + Smoking": "charges ~ bmi + C(smoker)",
    "BMI x Smoking": "charges ~ bmi * C(smoker)",
}

results = []
trained_models = {}
for name, formula in formulas.items():
    model = smf.ols(formula, data=train_df).fit()
    trained_models[name] = model
    predictions = model.predict(test_df)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(test_df["charges"], predictions),
        "RMSE": np.sqrt(mean_squared_error(test_df["charges"], predictions)),
        "Test R-squared": r2_score(test_df["charges"], predictions),
    })

evaluation = pd.DataFrame(results)
print("\n--- Test Set Model Evaluation ---")
print(evaluation.round(4).to_string(index=False))
evaluation.to_csv(PROJECT_DIR / "model_evaluation.csv", index=False)

# 8. Plot actual vs. predicted values from the TRAINED interaction model.
y_actual = test_df["charges"]
y_pred = trained_models["BMI x Smoking"].predict(test_df)
min_value = min(y_actual.min(), y_pred.min())
max_value = max(y_actual.max(), y_pred.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_actual, y_pred, alpha=0.5)
plt.plot(
    [min_value, max_value], [min_value, max_value],
    "k--", label="Perfect Prediction"
)
plt.xlabel("Actual Medical Charges ($)")
plt.ylabel("Predicted Medical Charges ($)")
plt.title("Actual vs. Predicted Medical Charges")
plt.legend()
plt.tight_layout()
plt.savefig(PROJECT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

print("\nAll analysis tables and figures saved in:", PROJECT_DIR)
