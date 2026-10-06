# House Price Prediction Using Machine Learning

## Project Overview

This project predicts house sale prices using machine learning regression models. It compares Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting to identify a model that performs well on unseen validation data.

The project also evaluates a log-transformed target, performs five-fold cross-validation on the selected Ridge Regression configuration, and saves a trained model for reuse.

## Objectives

- Explore and inspect the cleaned house-price dataset.
- Train and compare multiple regression models.
- Evaluate models using MAE, RMSE, and R².
- Investigate the effect of applying a `log1p` transformation to the target.
- Validate the selected model using five-fold cross-validation.
- Save the trained model and generate predictions for the test dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Dataset

The project uses two cleaned CSV files:

- `Cleaned train.csv` — training features and the `Saleprice` target.
- `Cleaned test.csv` — features used to generate predictions.

The training dataset contains **1,458 rows and 380 columns**, including the target column. The test dataset contains **1,459 rows and 379 columns**.

The target variable is `Saleprice`.

The project excludes the `Id` column from model training and removes four constant features:

- `has2ndfloor`
- `hasgarage`
- `hasbsmt`
- `MSSubClass_150`

After these exclusions, the model uses 374 features.

*Note: The cleaned CSV files are the inputs used in this project. Refer to the original dataset source and its license before redistributing the data.*

## Project Workflow

1. **Data inspection:** Examine dataset dimensions, data types, target distribution, and feature characteristics.
2. **Preprocessing:** Exclude the identifier and constant features, and align the training and test feature columns.
3. **Model training:** Train Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting models.
4. **Target transformation:** Compare models trained on the original target with models trained on `log1p(Saleprice)`.
5. **Model evaluation:** Compare MAE, RMSE, and R² on a holdout validation set.
6. **Cross-validation:** Evaluate the selected Ridge Regression model with a log-transformed target using five folds.
7. **Model persistence:** Save the selected model and its associated feature information using Joblib.
8. **Prediction:** Reload the saved model and generate predictions for the test dataset.

## Model Comparison

The following results were obtained on the holdout validation set.

| Model | Target | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Ridge Regression | `log1p` | 14,374.27 | 20,618.04 | 0.9230 |
| Gradient Boosting | `log1p` | 15,379.69 | 21,048.30 | 0.9198 |
| Gradient Boosting | Original | 16,633.31 | 22,972.05 | 0.9045 |
| Random Forest | Original | 16,189.08 | 23,667.06 | 0.8986 |
| Random Forest | `log1p` | 16,082.40 | 24,035.51 | 0.8954 |
| Ridge Regression | Original | 18,067.82 | 25,465.47 | 0.8826 |
| Linear Regression | `log1p` | 16,198.40 | 27,023.36 | 0.8678 |
| Linear Regression | Original | 20,090.90 | 29,810.62 | 0.8391 |

**Selected configuration:** Ridge Regression with a `log1p`-transformed target.

The selection is based on the lowest RMSE among the configurations evaluated on the holdout validation set.

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly.
- **R² (Coefficient of Determination):** Measures how much variation in the target is explained by the model relative to a baseline.

The reported error metrics use the target's original scale after transforming predictions back from the logarithmic scale. Their units are the same as those of `Saleprice`.

## Five-Fold Cross-Validation

The selected Ridge Regression configuration was evaluated using five-fold cross-validation.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| MAE | 14,516.59 | 944.06 |
| RMSE | 22,401.29 | 2,552.19 |
| R² | 0.9203 | 0.0091 |

The cross-validation results indicate relatively consistent R² scores across the five folds. These results apply to the selected Ridge Regression configuration; the other model types were not compared using the same five-fold procedure in this experiment.

## Project Structure

```text
House Price Prediction/
├── data/
│   ├── Cleaned train.csv
│   └── Cleaned test.csv
├── models/
│   └── house_price_model.joblib
├── outputs/
│   ├── model_comparison.csv
│   ├── predictions.csv
│   ├── cross_validation_results.csv
│   └── predictions_reloaded_model.csv
├── src/
│   ├── inspect_target.py
│   ├── house_price_prediction.py
│   ├── validate_model.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AnuLily02/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to Run

Run the scripts from the project root directory.

### Inspect the target

```bash
python src/inspect_target.py
```

### Train and compare models

```bash
python src/house_price_prediction.py
```

### Run five-fold cross-validation

```bash
python src/validate_model.py
```

### Reload the saved model and generate predictions

```bash
python src/predict.py
```

The prediction script checks the feature columns, loads the saved model, converts predictions back from the log scale when required, and saves the results to `outputs/predictions_reloaded_model.csv`.

## Outputs

The project generates the following files:

- `outputs/model_comparison.csv` — holdout validation metrics for the evaluated models.
- `outputs/cross_validation_results.csv` — five-fold validation results.
- `outputs/predictions.csv` — predictions generated during model training and evaluation.
- `outputs/predictions_reloaded_model.csv` — predictions generated by reloading the saved model.
- `models/house_price_model.joblib` — serialized model bundle used for inference.

## Limitations and Future Improvements

- Investigate extreme predictions and assess whether the model extrapolates beyond the range of observed sale prices.
- Compare all candidate models using the same cross-validation strategy.
- Tune hyperparameters using cross-validation.
- Evaluate residuals and prediction errors across different house-price ranges.
- Investigate prediction bias introduced by transforming predictions back from the logarithmic scale.
- Add visualizations for target distribution, model comparison, and actual versus predicted prices.
- Document the original dataset source and license.

## Conclusion

This project demonstrates an end-to-end machine learning regression workflow, from inspecting a cleaned dataset and comparing candidate models to cross-validating a selected configuration, saving the model, and generating reusable predictions.

Ridge Regression with a log-transformed target achieved the strongest holdout RMSE among the tested configurations. The results provide a baseline for further model validation and improvement.

## Author

**Anupama Balakrishnan**

GitHub: [AnuLily02](https://github.com/AnuLily02)
