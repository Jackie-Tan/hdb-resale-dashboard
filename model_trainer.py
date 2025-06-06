import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('Resale flat prices based on registration date from Jan-2017 onwards.csv')

# Preprocessing
df['month'] = pd.to_datetime(df['month'])
df['resale_year'] = df['month'].dt.year
df['remaining_lease'] = (df['lease_commence_date'] + 99) - df['resale_year']
df.to_csv('cleaned_hdb_resale_price_data.csv', index=False)

# Select features
features = ['town', 'flat_type', 'floor_area_sqm', 'remaining_lease', 'resale_year', 'storey_range', 'flat_model']
X = df[features]
y = df['resale_price']

# Define preprocessing and model pipeline
categorical = ['town', 'flat_type', 'storey_range', 'flat_model']
numerical = ['floor_area_sqm', 'remaining_lease', 'resale_year']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
])

model.fit(X, y)

# Save the model
joblib.dump(model, 'hdb_resale_price_model.pkl')