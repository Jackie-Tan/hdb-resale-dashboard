import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load data (replace with actual data path if needed)
df = pd.read_csv("Resale flat prices based on registration date from Jan-2017 onwards.csv")

# Preprocessing
st.title("HDB Resale Price Predictor")
df['month'] = pd.to_datetime(df['month'])
df['resale_year'] = df['month'].dt.year
df['remaining_lease'] = (df['lease_commence_date'] + 99) - df['resale_year']

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

# User inputs
town = st.selectbox("Select Town", df['town'].unique())
flat_type = st.selectbox("Select Flat Type", df['flat_type'].unique())
floor_area = st.slider("Floor Area (sqm)", 30, 200, 90)
storey_range = st.selectbox("Select Storey Range", df['storey_range'].unique())
flat_model = st.selectbox("Select Flat Model", df['flat_model'].unique())
resale_year = st.slider("Resale Year", int(df['resale_year'].min()), int(df['resale_year'].max()), 2024)
lease_commence_date = st.slider("Lease Commence Year", 1966, 2023, 2005)

# Calculate remaining lease
remaining_lease = (lease_commence_date + 99) - resale_year

# Predict
input_df = pd.DataFrame([{ 
    'town': town,
    'flat_type': flat_type,
    'floor_area_sqm': floor_area,
    'remaining_lease': remaining_lease,
    'resale_year': resale_year,
    'storey_range': storey_range,
    'flat_model': flat_model
}])

prediction = model.predict(input_df)[0]
st.subheader(f"Predicted Resale Price: SGD {int(prediction):,}")

# Historical trend chart
st.markdown("---")
st.subheader("Historical Price Trends")

# Filtered trend data
trend_data = df[(df['town'] == town) & (df['flat_type'] == flat_type)]
monthly_avg = trend_data.groupby('month')['resale_price'].mean().reset_index()

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly_avg['month'], monthly_avg['resale_price'], marker='o')
ax.set_title(f"Avg Resale Price in {town} ({flat_type})")
ax.set_xlabel("Month")
ax.set_ylabel("Average Resale Price (SGD)")
ax.grid(True)

st.pyplot(fig)
