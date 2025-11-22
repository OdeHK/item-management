import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
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

# Default specification limits based on product item
spec_defaults = {
    'Perpendicularity': {'LSL': 0.0, 'USL': 0.5, 'Target': 0.25},
    'Perpendicularity ': {'LSL': 0.0, 'USL': 0.5, 'Target': 0.25},  # With trailing space
    'M-Length': {'LSL': 1614.0, 'USL': 1615.0, 'Target': 1614.5},
    'M-Width': {'LSL': 324.25, 'USL': 324.5, 'Target': 324.375},
    'M-Thickness': {'LSL': 0.48, 'USL': 0.52, 'Target': 0.5},
    'Tensilon - Bottom': {'LSL': 4200.0, 'USL': None, 'Target': None},
    'Tensilon - Top': {'LSL': 4200.0, 'USL': None, 'Target': None}
}

# Get defaults for selected item
defaults = spec_defaults.get(selected, {'LSL': 0.0, 'USL': 10.0, 'Target': 5.0})

# Specification limits inputs
st.sidebar.markdown("---")
st.sidebar.header("Cấu hình giới hạn")

lsl = st.sidebar.number_input(
    "LSL (Lower Spec Limit)", 
    value=defaults['LSL'], 
    format="%.4f"
)

# Handle None for USL
if defaults['USL'] is not None:
    usl = st.sidebar.number_input(
        "USL (Upper Spec Limit)", 
        value=defaults['USL'], 
        format="%.4f"
    )
else:
    usl_enabled = st.sidebar.checkbox("Enable USL", value=False)
    if usl_enabled:
        usl = st.sidebar.number_input(
            "USL (Upper Spec Limit)", 
            value=10.0, 
            format="%.4f"
        )
    else:
        usl = None

# Handle None for Target
if defaults['Target'] is not None:
    target = st.sidebar.number_input(
        "Target", 
        value=defaults['Target'], 
        format="%.4f"
    )
else:
    target_enabled = st.sidebar.checkbox("Enable Target", value=False)
    if target_enabled:
        target = st.sidebar.number_input(
            "Target", 
            value=5.0, 
            format="%.4f"
        )
    else:
        target = None

st.markdown("# Chart Visualization")

# I-MR Chart
st.markdown("## I-MR Chart (Individual and Moving Range)")
create_imr_chart(item_df)

# SPC Rule Violations (from app_with_excel.py)
st.markdown("### SPC Rule Violations")

def detect_spc_violations(values, mean, ucl, lcl):
    """
    Detect SPC violations based on Western Electric Rules
    """
    violations = []
    n = len(values)
    sigma = (ucl - mean) / 3 if ucl > mean else 0
    
    if sigma == 0:
        return ["Cannot calculate - insufficient data"]
    
    # Rule 1: 1 point beyond 3σ
    out_of_control = (values > ucl) | (values < lcl)
    if np.any(out_of_control):
        out_indices = np.where(out_of_control)[0] + 1
        violations.append(f"Rule 1: Điểm ngoài 3σ tại observation {list(out_indices)}")
    
    # Rule 2: 2/3 points > 2σ (same side)
    upper_2sigma = mean + 2*sigma
    lower_2sigma = mean - 2*sigma
    
    for i in range(n-2):
        # Check upper side
        if (values[i] > upper_2sigma and 
            values[i+1] > upper_2sigma and 
            values[i+2] > upper_2sigma):
            violations.append(f"Rule 2: 2/3 điểm > 2σ (phía trên) tại {i+1}-{i+3}")
            break
    
    for i in range(n-2):
        # Check lower side
        if (values[i] < lower_2sigma and 
            values[i+1] < lower_2sigma and 
            values[i+2] < lower_2sigma):
            violations.append(f"Rule 2: 2/3 điểm > 2σ (phía dưới) tại {i+1}-{i+3}")
            break
    
    # Rule 4: 8 points same side of center line
    for i in range(n-7):
        if all(v > mean for v in values[i:i+8]):
            violations.append(f"Rule 4: 8 điểm cùng phía (trên) tại {i+1}-{i+8}")
            break
        elif all(v < mean for v in values[i:i+8]):
            violations.append(f"Rule 4: 8 điểm cùng phía (dưới) tại {i+1}-{i+8}")
            break
    
    # Rule 5: 6 points increasing or decreasing
    for i in range(n-5):
        if all(values[j] < values[j+1] for j in range(i, i+5)):
            violations.append(f"Rule 5: 6 điểm tăng liên tiếp tại {i+1}-{i+6}")
            break
        elif all(values[j] > values[j+1] for j in range(i, i+5)):
            violations.append(f"Rule 5: 6 điểm giảm liên tiếp tại {i+1}-{i+6}")
            break
    
    return violations if violations else ["✓ Không vi phạm quy tắc SPC"]

# Calculate control limits for SPC rules
values = item_df['Measure'].values
if len(values) >= 2:
    mean = np.mean(values)
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)
    d2 = 1.128
    sigma = mr_bar / d2
    ucl_i = mean + 3 * sigma
    lcl_i = max(0, mean - 3 * sigma)
    
    violations = detect_spc_violations(values, mean, ucl_i, lcl_i)
    
    for v in violations:
        if "✓" in v or "Không" in v.lower():
            st.success(v)
        else:
            st.warning(v)
else:
    st.info("Cần ít nhất 2 điểm dữ liệu để phân tích SPC")

# Display data summary
st.markdown("### Dữ liệu đo")
display_df = item_df[['Date', 'Measure']].copy()
if len(values) >= 2:
    display_df['Status'] = display_df['Measure'].apply(
        lambda x: 'Out' if x > ucl_i or x < lcl_i else 'In'
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng mẫu", len(values))
    with col2:
        in_control = np.sum((values >= lcl_i) & (values <= ucl_i))
        st.metric("In Control", f"{in_control}/{len(values)}")
    with col3:
        out_control = np.sum((values > ucl_i) | (values < lcl_i))
        st.metric("Out Control", f"{out_control}/{len(values)}")

st.dataframe(display_df, use_container_width=True, height=300)

# Histogram with Normal Curve and Probability Plot
st.markdown("## Histogram with Normal Distribution")
create_histogram_with_probability(item_df, LSL=lsl, USL=usl, Target=target, auto_spec_limits=False)