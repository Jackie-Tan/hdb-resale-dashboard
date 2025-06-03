import streamlit as st
import pandas as pd
import joblib
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates


# Load data and cache
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data('cleaned_hdb_resale_price_data.csv')

st.title("HDB Resale Price Predictor")
# User inputs
town = st.selectbox("Select Town", df['town'].unique())
flat_type = st.selectbox("Select Flat Type", df['flat_type'].unique())
floor_area = st.slider("Floor Area (sqm)", 30, 200, 90)
storey_range = st.selectbox("Select Storey Range", df['storey_range'].unique())
flat_model = st.selectbox("Select Flat Model", df['flat_model'].unique())
resale_year = st.slider("Resale Year", int(df['resale_year'].min()), int(df['resale_year'].max()), 2025)
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

# Load model and cache
@st.cache_resource
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model('hdb_resale_price_model.pkl')

# Make prediction and cache
@st.cache_data
def run_model(input_df):
    prediction = model.predict(input_df)[0]
    return prediction

prediction = run_model(input_df)
st.subheader(f"Predicted Resale Price: SGD {int(prediction):,}")

# Historical trend chart
st.markdown("---")
st.subheader("Historical Price Trends")

# Filtered trend data
# trend_data = df[(df['town'] == town) & (df['flat_type'] == flat_type)].copy()
trend_data = df[(df['town'] == town) & (df['flat_type'] == flat_type)].copy()
trend_data['month'] = pd.to_datetime(trend_data['month'], errors='coerce')
trend_data = trend_data.dropna(subset=['month'])
monthly_avg = trend_data.groupby('month')['resale_price'].mean().reset_index()

# Streamlit line chart
st.subheader(f"Avg Resale Price in {town} ({flat_type})")
st.scatter_chart(monthly_avg.set_index('month')['resale_price'])

# # Filtered trend data
# trend_data = df[(df['town'] == town) & (df['flat_type'] == flat_type)]
# monthly_avg = trend_data.groupby('month')['resale_price'].mean().reset_index()

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(monthly_avg['month'], monthly_avg['resale_price'], marker='o')

# # Clean and readable x-axis formatting
# ax.xaxis.set_major_locator(mdates.YearLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# fig.autofmt_xdate()  # Auto-rotate if needed

# ax.set_title(f"Avg Resale Price in {town} ({flat_type})")
# ax.set_xlabel("Month")
# ax.set_ylabel("Average Resale Price (SGD)")
# ax.grid(True)

# st.pyplot(fig)


