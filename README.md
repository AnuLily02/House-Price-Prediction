# House Price Prediction

Predicts house prices with machine learning (scikit-learn). Several regressors are compared, the best one is retrained on all data and saved, then used to predict on the test set.

**Dataset:** [House Price Prediction – Cleaned Dataset (Kaggle)](https://www.kaggle.com/datasets/chandramoulinaidu/house-price-prediction-cleaned-dataset)

## Structure
```
data/        train.csv, test.csv (put the Kaggle files here)
models/      house_price_model.pkl (generated)
outputs/     predictions.csv, model_comparison.csv (generated)
src/         house_price_prediction.py
```

## Setup
```bash
pip install -r requirements.txt
```
Download the dataset from Kaggle and save the cleaned train file as `data/train.csv` and the cleaned test file as `data/test.csv`.

## Run
```bash
python src/house_price_prediction.py
# if the target column isn't auto-detected:
python src/house_price_prediction.py --target SalePrice
```

## Pipeline
1. Load train data, auto-detect the target column
2. Impute missing values, scale numeric features, one-hot encode categoricals
3. Log-transform the target (when all prices are positive)
4. Compare Linear, Ridge, Lasso, Random Forest, Gradient Boosting (hold-out RMSE/MAE/R² + 5-fold CV R²)
5. Refit the best model on all training data and save to `models/`
6. Predict on the test set and write `outputs/predictions.csv`
