import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Property Price Analyzer",
    page_icon="🏠",
)

# Slider styling: bright orange for dark mode
st.markdown(
    """
<style>
[data-baseweb="slider"] .rc-slider-track {
    background-color: #FFA500 !important;
}
[data-baseweb="slider"] .rc-slider-handle {
    border-color: #FFA500 !important;
    background-color: #FFA500 !important;
}
[data-baseweb="slider"] .rc-slider-handle:focus,
[data-baseweb="slider"] .rc-slider-handle:hover {
    border-color: #FFA500 !important;
    box-shadow: 0 0 5px #FFA500 !important;
}
[data-baseweb="slider"] .rc-slider-dot-active {
    border-color: #FFA500 !important;
    background-color: #FFA500 !important;
}
</style>
    """,
    unsafe_allow_html=True,
)

# Load data and cache
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data('cleaned_hdb_resale_price_data.csv')

st.title("Property Price Analyzer")
# User inputs
raw_towns = df['town'].unique().tolist()
town_pairs = sorted([(t.title(), t) for t in raw_towns], key=lambda x: x[0])
placeholder = "Select a Town"
town = st.selectbox("Select Town", [placeholder] + [d for d, _ in town_pairs])
if town == placeholder:
    st.warning("Please select a Town to continue.")
    st.stop()
raw_town = dict(town_pairs)[town]
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
    'town': raw_town,
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
trend_data = df[(df['town'] == raw_town) & (df['flat_type'] == flat_type)].copy()
trend_data['month'] = pd.to_datetime(trend_data['month'], errors='coerce')
trend_data = trend_data.dropna(subset=['month'])
monthly_avg = trend_data.groupby('month')['resale_price'].mean().reset_index()

# Streamlit line chart
st.subheader(f"Avg Resale Price in {town} ({flat_type})")
st.scatter_chart(monthly_avg.set_index('month')['resale_price'])