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
placeholder_ft = "Select a Flat Type"
raw_flat_types = df[df['town'] == raw_town]['flat_type'].unique().tolist()
flat_type_pairs = sorted([(ft.title(), ft) for ft in raw_flat_types], key=lambda x: x[0])
flat_type_display = st.selectbox(
    "Select Flat Type", [placeholder_ft] + [d for d, _ in flat_type_pairs]
)
if flat_type_display == placeholder_ft:
    st.warning("Please select a Flat Type to continue.")
    st.stop()
flat_type = dict(flat_type_pairs)[flat_type_display]

placeholder_fm = "Select a Flat Model"
raw_flat_models = df[(df["town"] == raw_town) & (df["flat_type"] == flat_type)]["flat_model"].unique().tolist()
flat_models_for_type = sorted(raw_flat_models)
flat_model = st.selectbox("Select Flat Model", [placeholder_fm] + flat_models_for_type)
if flat_model == placeholder_fm:
    st.warning("Please select a Flat Model to continue.")
    st.stop()

subset_fa = df[
    (df["town"] == raw_town)
    & (df["flat_type"] == flat_type)
    & (df["flat_model"] == flat_model)
]
unique_fas = sorted(int(x) for x in subset_fa["floor_area_sqm"].unique())
default_fa = int(subset_fa["floor_area_sqm"].median())

if len(unique_fas) == 1:
    st.info(f"Only one floor area available: {unique_fas[0]} sqm")
    floor_area = unique_fas[0]
else:
    default_index = unique_fas.index(default_fa) if default_fa in unique_fas else 0
    floor_area = st.selectbox(
        "Floor Area (sqm)",
        unique_fas,
        index=default_index,
    )

raw_storeys = subset_fa["storey_range"].unique().tolist()
sorted_storeys = sorted(raw_storeys, key=lambda x: int(x.split()[0]))
display_storeys = [s.lower() for s in sorted_storeys]
mapping_storeys = dict(zip(display_storeys, sorted_storeys))
default_sr_index = len(display_storeys) // 2

if len(display_storeys) == 1:
    st.info(f"Only one storey range available: {display_storeys[0]}")
    storey_range = sorted_storeys[0]
else:
    selected_storey = st.selectbox(
        "Select Storey Range",
        display_storeys,
        index=default_sr_index,
    )
    storey_range = mapping_storeys[selected_storey]
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
st.subheader(f"Avg Resale Price in {town} ({flat_type_display})")
st.scatter_chart(monthly_avg.set_index('month')['resale_price'])