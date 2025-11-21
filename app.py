import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from charts.imr_chart import create_imr_chart
from charts.histogram_chart import create_histogram_with_probability

st.set_page_config(page_title="Item Management Dashboard", layout="wide")

st.title("Item Management Dashboard")

# Connect to Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(worksheet="Measure")

# Sidebar selection
items = sorted(df['ProductItem'].unique())
selected = st.sidebar.selectbox("Chọn Product Item", items)
item_df = df[df['ProductItem'] == selected].reset_index(drop=True)

st.sidebar.dataframe(item_df[['Date', 'Measure']], hide_index=True)

st.markdown("# Chart Visualization")

# I-MR Chart
st.markdown("## I-MR Chart (Individual and Moving Range)")
create_imr_chart(item_df)

# Histogram with Normal Curve and Probability Plot
st.markdown("## Histogram with Normal Distribution")
create_histogram_with_probability(item_df)
