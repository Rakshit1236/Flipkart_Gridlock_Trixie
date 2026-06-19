import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Datasets
print("Loading data...")
train_df = pd.read_csv('DATASET/train.csv')
test_df = pd.read_csv('DATASET/test.csv')

# 2. Separate target (y) and features (X)
X_train = train_df.drop(['demand', 'Index'], axis=1)
y_train = train_df['demand']
X_test = test_df.drop(['Index'], axis=1)
test_indices = test_df['Index']

# 3. Safe Feature Engineering
def extract_time(df):
    df_temp = df.copy()
    # Split "HH:MM" into hours and minutes
    df_temp[['hour', 'minute']] = df_temp['timestamp'].str.split(':', expand=True).astype(float)
    # Add a continuous time feature
    df_temp['time_in_mins'] = df_temp['hour'] * 60 + df_temp['minute']
    return df_temp.drop('timestamp', axis=1)

X_train = extract_time(X_train)
X_test = extract_time(X_test)

# 4. Define Column Groupings
cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
num_cols = ['day', 'NumberofLanes', 'Temperature', 'hour', 'minute', 'time_in_mins']

# 5. Build Preprocessing Pipelines
num_transformer = SimpleImputer(strategy='median')
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# 6. Build the TUNED Random Forest Pipeline
# These parameters constrain the trees so they generalize better to the test set
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=300,        # 3x more trees than the baseline for better averaging
        max_depth=15,            # Prevents trees from growing too deep and memorizing noise
        min_samples_split=5,     # Requires at least 5 samples to create a new branch
        min_samples_leaf=2,      # Ensures leaf nodes have at least 2 samples
        random_state=42, 
        n_jobs=-1
    ))
])

# 7. Train the Model
print("Training the Tuned Random Forest model... (This will take a little longer)")
model.fit(X_train, y_train)

# 8. Generate Predictions
print("Generating predictions for the test set...")
predictions = model.predict(X_test)

# 9. Format and Save Output
submission = pd.DataFrame({
    'Index': test_indices,
    'demand': predictions
})

submission.to_csv('tuned_rf_submission.csv', index=False)
print("Success! Predictions saved to 'tuned_rf_submission.csv'")