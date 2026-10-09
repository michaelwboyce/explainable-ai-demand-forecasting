# Explainable AI for Demand Forecasting

**A Python machine learning interpretability project comparing how a random forest and a k-nearest neighbors (KNN) regressor use customer purchase history to predict next-month home-decor sales.**

## Project overview

Forecasts are more useful when stakeholders understand the factors behind them. In this project, I analyzed two pretrained sales-forecasting models with **SHAP (SHapley Additive exPlanations)**, identified their most influential input features, and compared their global feature-importance patterns using cosine similarity.

The target is `nextmonth__home_decor`, the log-transformed value of a customer's next-month home-decor sales (`log(sales + 1)`). The supplied predictors include historical sales, lagged values, moving averages, customer tenure, and sales in other product categories.

## Methods

1. **Random forest:** Use `shap.TreeExplainer` on a 500-row test sample.
2. **KNN:** Use `shap.KernelExplainer` with a 10-row training background and a 50-row test sample. An approximation budget of `nsamples=120` makes the calculation more practical.
3. **Global importance:** Compute mean absolute SHAP values for each feature.
4. **Comparison:** Use cosine similarity between the **full, feature-aligned** importance vectors of the two models. This measures agreement in the *relative importance patterns*, not prediction accuracy.
5. **Interpretation:** Use the score to discuss consistency of the explanations, without treating consistency alone as proof of model correctness.

## Results

The two feature-importance vectors achieved **cosine similarity = 0.94** in the DataCamp project environment, indicating strong agreement in the models' overall importance patterns.

| Random forest: top five | Mean absolute SHAP | KNN: top five | Mean absolute SHAP |
|---|---:|---|---:|
| `lag2` | 0.117177 | `lag2` | 0.192028 |
| `sma_4m` | 0.098794 | `lag1` | 0.153445 |
| `lag1` | 0.095220 | `sma_2m` | 0.101085 |
| `sma_2m` | 0.059562 | `sma_4m` | 0.088489 |
| `sma_2m__colourful_essentials` | 0.058533 | `home_decor` | 0.049619 |

**What stands out:** Both models place `lag2` first, and four of their top-five features overlap. Historical sales and moving averages appear central to the forecast. A high consistency score does **not** establish predictive accuracy, causal relationships, or reliability on future data.

The script uses an **illustrative 0.80 cutoff** to create the assignment's required `reliable` (`"yes"` or `"no"`) variable. This threshold is a project interpretation heuristic, **not** a statistically validated threshold or a DataCamp-specified grading rule.

## Technologies

Python 3.10 · pandas · NumPy · scikit-learn · SHAP · joblib

## Run the code

1. Install Python 3.10 and dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Obtain the project data and trained model files from the original DataCamp exercise and place them in a local `data/` directory:

   ```text
   data/
     X_train.csv
     X_test.csv
     y_train.csv
     y_test.csv
     model.pkl
     knn_model.pkl
   ```

3. Run:

   ```bash
   python demystifying_ai_demand_forecasting.py
   ```

The repository intentionally **does not redistribute the exercise's datasets or pretrained models**, which may be subject to third-party usage terms. Reproducing results also depends on compatibility with the packages used to serialize the supplied models. Load `.pkl` models only from a source you trust.

## Limitations and next steps

- KNN explanations are approximations estimated from a small sample; the importance rankings can change with the background, samples, and computational budget.
- SHAP values describe fitted-model predictions, not necessarily causal effects.
- Proper validation of sales-forecast accuracy requires held-out performance metrics and temporal evaluation.
- Potential improvements: compare out-of-sample MAE/RMSE, create SHAP summary plots, and test explanation stability across multiple samples.

## Acknowledgment

This is an educational portfolio implementation based on DataCamp's **“Demystifying AI: Explaining Demand Forecasting Models”** project. The findings reflect the supplied exercise models and a sampled dataset.
