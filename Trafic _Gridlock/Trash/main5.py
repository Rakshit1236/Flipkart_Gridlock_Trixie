import os
import numpy as np
import pandas as pd
import pygeohash as Geohash
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def load_and_preprocess(filepath):
    '''Loads the dataset and performs feature engineering on time and geohash.'''
    df = pd.read_csv(filepath)
    
    # Handle both geohash and geohash6 column names
    if 'geohash6' in df.columns:
        df['geohash_col'] = df['geohash6']
    elif 'geohash' in df.columns:
        df['geohash_col'] = df['geohash']
    else:
        raise ValueError("DataFrame must contain either 'geohash6' or 'geohash' column")
    
    df['hours'] = df['timestamp'].map(lambda x: int(x.split(':')[0]))
    df['mins'] = df['timestamp'].map(lambda x: int(x.split(':')[1]))
    # Convert day, hours, mins into a single continuous time feature (in minutes)
    df['time'] = 24 * 60 * (df['day'] - 1) + 60 * df['hours'] + df['mins']
    
    # Decode geohash into Latitude and Longitude
    df['Latitude'] = df.geohash_col.map(lambda x: float(Geohash.decode_exactly(x)[0]))
    df['Longitude'] = df.geohash_col.map(lambda x: float(Geohash.decode_exactly(x)[1]))
    
    # Sort chronologically and spatially to prevent data leakage in TS splits
    df = df.sort_values(by=['time', 'Latitude', 'Longitude'], ascending=True)
    df = df.reset_index(drop=True)
    return df

def convert_time(time):
    '''Reverses the continuous time feature back to day, hour, minute, and timestamp string.'''
    day = int(time / (24 * 60)) + 1
    hour = int((time - (day - 1) * 24 * 60) / 60)
    minute = time - (day - 1) * 24 * 60 - hour * 60
    timestamp = ':'.join((str(hour), str(minute)))
    return (day, hour, minute, timestamp)

def main():
    # ---------------------------------------------------------
    # Part 1: Data Loading & Preprocessing
    # ---------------------------------------------------------
    print("Loading training data...")
    # Update this path if your training data is located elsewhere
    train_filepath = 'training.csv'
    
    if not os.path.exists(train_filepath):
        print(f"Warning: {train_filepath} not found. Please ensure the data is in the correct directory.")
        # We'll skip execution below if data isn't present, but structure remains intact.
        return
        
    df_train = load_and_preprocess(train_filepath)
    
    print("Splitting data into train and test (Chronological)...")
    max_day = df_train.day.max()
    max_time = df_train.time.max()
    
    # Extracting the last 14 days as done in the original notebook
    train_start = df_train[df_train.day == 61 - 13].index[0]
    test_start = df_train[df_train.time == max_time - 15 * 4].index[0]
    
    Xtrain = df_train[['time', 'Latitude', 'Longitude']].iloc[train_start:test_start, :]
    Xtest = df_train[['time', 'Latitude', 'Longitude']].iloc[test_start:, :]
    ytrain = df_train.demand.iloc[train_start:test_start]
    ytest = df_train.demand.iloc[test_start:]

    # ---------------------------------------------------------
    # Part 2: Hyperparameter Tuning
    # ---------------------------------------------------------
    print("Starting Hyperparameter Tuning...")
    param_dist = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [5, 10, 15, 20],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5]
    }
    
    # Using TimeSeriesSplit to validate time-series data correctly
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_dist, 
        n_iter=10,                      
        scoring='neg_root_mean_squared_error',
        cv=tscv, 
        verbose=2, 
        random_state=42, 
        n_jobs=-1
    )
    
    # We train on a subset (last 100,000 rows of train) to keep tuning time reasonable
    print("Fitting RandomizedSearchCV on recent subset of training data...")
    subset_size = 100000
    random_search.fit(Xtrain.iloc[-subset_size:], ytrain.iloc[-subset_size:])
    
    best_params = random_search.best_params_
    print("\nBest Parameters Found:")
    print(best_params)
    print(f"Best Cross-Validated RMSE: {-random_search.best_score_}")
    
    best_xgb = random_search.best_estimator_
    ytest_pred_tuned = best_xgb.predict(Xtest)
    tuned_rmse = np.sqrt(mean_squared_error(ytest, ytest_pred_tuned))
    print(f"Tuned Test RMSE (on held-out data): {tuned_rmse}")

    # ---------------------------------------------------------
    # Part 3: Demand Prediction (T+1 to T+5)
    # ---------------------------------------------------------
    def predict5ts(link, model_params, test_df):
        '''
        Predicts demand for T+1 to T+5 for geohashes in the test dataset.
        '''
        print(f"\nProcessing test dataset: {link}")
        df = load_and_preprocess(link)
        
        X = df[['time', 'Latitude', 'Longitude']]
        
        # Check if demand column exists (for training-like data)
        if 'demand' in df.columns:
            y = df.demand
        else:
            # For test data without demand, we'll fit on the full training data again
            print("No demand column in test data. Using full training data for model fitting...")
            X = df_train[['time', 'Latitude', 'Longitude']]
            y = df_train.demand
        
        # Train the model with the best found parameters
        print("Training final prediction model...")
        final_model = XGBRegressor(**model_params, objective='reg:squarederror', random_state=42)
        final_model.fit(X, y)
        
        T = df.time.max()
        future_times = [T + 15 * i for i in range(1, 6)]
        
        # Get unique geohashes from test data
        test_geohashes = test_df['geohash'].unique() if 'geohash' in test_df.columns else test_df['geohash6'].unique()
        
        print(f"Generating predictions for T+1 to T+5 ({len(test_geohashes)} geohashes)...")
        results = []
        count = 0
        for t in future_times:
            for gh in test_geohashes:
                try:
                    lat, lon = Geohash.decode_exactly(gh)
                    day, hour, minute, timestamp = convert_time(int(t))
                    results.append({
                        'geohash6': gh,
                        'day': int(day),
                        'timestamp': str(timestamp),
                        'time': int(t),
                        'Latitude': float(lat),
                        'Longitude': float(lon)
                    })
                    count += 1
                except Exception as e:
                    # Skip invalid geohashes
                    pass
        
        print(f"Successfully prepared {count} predictions")
        
        if len(results) == 0:
            print("Error: No valid predictions could be generated")
            return
            
        df_pred = pd.DataFrame(results)
        print(f"DataFrame created with columns: {list(df_pred.columns)}")
        
        # Ensure the required columns exist
        if not all(col in df_pred.columns for col in ['time', 'Latitude', 'Longitude']):
            print(f"Error: Missing required columns. Available columns: {list(df_pred.columns)}")
            return
            
        X_future = df_pred[['time', 'Latitude', 'Longitude']]
        
        # Make predictions
        print("Making predictions...")
        df_pred['demand'] = final_model.predict(X_future)
        
        # Format output
        output = df_pred[['geohash6', 'day', 'timestamp', 'demand']]
        output.to_csv('output.csv', index=False)
        print(f"Predictions successfully saved to output.csv ({len(output)} predictions)")

    # ---------------------------------------------------------
    # Part 4: Execute final prediction
    # ---------------------------------------------------------
    print("\nGenerating predictions with best parameters...")
    test_df = pd.read_csv('DATASET/test.csv')
    test_link = 'DATASET/test.csv'
    predict5ts(link=test_link, model_params=best_params, test_df=test_df)
    
    print("\nModel training and prediction generation completed successfully!")

if __name__ == '__main__':
    main()