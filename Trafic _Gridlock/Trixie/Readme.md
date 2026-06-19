# Demand Prediction Model (MAIN888)

## Overview
This script trains a robust machine learning pipeline to predict demand based on geospatial, temporal, and environmental factors. It utilizes a multi-model ensemble approach (LightGBM, XGBoost, and CatBoost) with repeated cross-validation across multiple random seeds to ensure high generalization and stability.

## Data Requirements
The script expects three datasets in the same directory:
* `train.csv`: The primary training data.
* `training.csv`: Supplementary historical data used for generating aggregate features.
* `test.csv`: The holdout dataset for final predictions.

## Pipeline Breakdown

### 1. Feature Engineering
The script enriches the base data with several types of features:
* **Temporal Features:** Extracts hours and minutes, calculates cyclical time features using sine/cosine transformations, and flags specific time periods (rush hour, morning, evening).
* **Geospatial & Aggregations:** Truncates `geohash` strings to create broader regional groupings. It calculates historical statistics (mean, median, standard deviation, count) grouped by geohash, hour, road type, and weather from the supplementary `training.csv` file.
* **Categorical Encoding:** Applies label encoding to features like Road Type, Weather, Landmarks, and Large Vehicles. 
* **Target Encoding:** Uses cross-validated target encoding (via `category_encoders`) on high-cardinality features (`geohash_5` and `geo4_hour`) to prevent data leakage.

### 2. Model Architecture
To predict demand, the target variable is transformed using `np.log1p` to handle skewness and predict on a logarithmic scale. The script uses three gradient boosting frameworks with pre-optimized hyperparameters:
* **LightGBM** (`LGBMRegressor`)
* **XGBoost** (`XGBRegressor`)
* **CatBoost** (`CatBoostRegressor`)

**Training Strategy:** It employs a 5-Fold Cross-Validation strategy, repeated across 5 different random seeds (resulting in 25 models per algorithm) to minimize variance and overfitting. 

### 3. Ensembling & Stacking
The pipeline evaluates predictions using the $R^2$ score (after reverting the log transformation with `np.expm1`). It combines the models using three strategies and automatically selects the best-performing one:
1.  **Weighted Average:** Uses Scipy's `minimize` (SLSQP method) to find the mathematically optimal weights for blending the LightGBM, XGBoost, and CatBoost outputs.
2.  **Stacking:** Trains a Ridge Regression meta-model on the Out-Of-Fold (OOF) predictions from the base models.
3.  **Blend:** A 50/50 combination of the Stacking output and the Weighted Average.

## Output
The script evaluates all ensemble methods, selects the approach with the highest Out-Of-Fold $R^2$ score, and generates the final predictions on the test set. 

The results are saved locally as `submission_888.csv`. Execution time and score metrics are printed dynamically to the console.