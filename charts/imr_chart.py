import streamlit as st
import altair as alt
import pandas as pd
import numpy as np


def calculate_imr_statistics(df):
    """
    Calculate I-MR chart statistics
    
    Parameters:
    df: DataFrame with 'Measure' column
    
    Returns:
    dict: Dictionary containing all statistics
    """
    # Individual values
    X = df['Measure'].values
    X_bar = np.mean(X)
    
    # Moving Range
    MR = np.abs(np.diff(X))
    MR_bar = np.mean(MR)
    
    # I Chart control limits
    UCL_I = X_bar + 2.66 * MR_bar
    LCL_I = X_bar - 2.66 * MR_bar
    
    # MR Chart control limits
    UCL_MR = 3.266 * MR_bar
    LCL_MR = 0
    
    return {
        'X_bar': X_bar,
        'MR_bar': MR_bar,
        'UCL_I': UCL_I,
        'LCL_I': LCL_I,
        'UCL_MR': UCL_MR,
        'LCL_MR': LCL_MR
    }


def calculate_smart_y_range(data_values, ucl, lcl):
    """
    Calculate smart Y-axis range that focuses on data region
    
    Parameters:
    data_values: Array of data points
    ucl: Upper control limit
    lcl: Lower control limit
    
    Returns:
    tuple: (y_min, y_max) for chart
    """
    # Get actual data range
    data_min = np.min(data_values)
    data_max = np.max(data_values)
    data_range = data_max - data_min
    
    # Control limits range
    control_range = ucl - lcl
    
    # If data range is very small compared to control range
    # Focus more on data region
    if data_range < control_range * 0.3:
        # Use data range with padding
        padding = data_range * 0.2
        y_min = max(lcl, data_min - padding)
        y_max = min(ucl, data_max + padding)
        
        # Ensure we still show control limits if they're close
        margin = control_range * 0.05
        if ucl - y_max < margin:
            y_max = ucl + margin
        if y_min - lcl < margin:
            y_min = lcl - margin
    else:
        # Data fills most of control range, use standard padding
        padding = control_range * 0.05
        y_min = lcl - padding
        y_max = ucl + padding
    
    return y_min, y_max


def create_individual_chart(df, stats):
    """
    Create Individual values chart (I Chart) with smart Y-axis scaling
    
    Parameters:
    df: DataFrame with measurement data
    stats: Dictionary of statistics
    
    Returns:
    Altair chart object
    """
    chart_df = df.copy()
    chart_df['Observation'] = range(1, len(chart_df) + 1)
    chart_df['Individual'] = chart_df['Measure']
    
    # Calculate smart Y range
    y_min, y_max = calculate_smart_y_range(
        chart_df['Individual'].values, 
        stats['UCL_I'], 
        stats['LCL_I']
    )
    
    base = alt.Chart(chart_df).encode(
        x=alt.X('Observation:Q', 
                title='Observation', 
                scale=alt.Scale(domain=[0, len(chart_df) + 1]))
    )
    
    # Individual values line
    line = base.mark_line(color='blue', point=True).encode(
        y=alt.Y('Individual:Q', 
                title='Individual values', 
                scale=alt.Scale(domain=[y_min, y_max]))
    )
    
    # Center Line
    cl = alt.Chart(pd.DataFrame({'y': [stats['X_bar']]})).mark_rule(
        color='green', strokeDash=[5, 5], size=2
    ).encode(y='y:Q')
    
    # UCL Line
    ucl = alt.Chart(pd.DataFrame({'y': [stats['UCL_I']]})).mark_rule(
        color='red', size=2
    ).encode(y='y:Q')
    
    # LCL Line
    lcl = alt.Chart(pd.DataFrame({'y': [stats['LCL_I']]})).mark_rule(
        color='red', size=2
    ).encode(y='y:Q')
    
    # Labels
    ucl_text = alt.Chart(pd.DataFrame({
        'x': [len(chart_df)], 
        'y': [stats['UCL_I']], 
        'text': [f"UCL: {stats['UCL_I']:.2f}"]
    })).mark_text(align='right', dx=40, dy=-5, fontSize=11).encode(
        x='x:Q', y='y:Q', text='text:N'
    )
    
    cl_text = alt.Chart(pd.DataFrame({
        'x': [len(chart_df)], 
        'y': [stats['X_bar']], 
        'text': [f"X̄: {stats['X_bar']:.2f}"]
    })).mark_text(align='right', dx=40, dy=-5, fontSize=11).encode(
        x='x:Q', y='y:Q', text='text:N'
    )
    
    lcl_text = alt.Chart(pd.DataFrame({
        'x': [len(chart_df)], 
        'y': [stats['LCL_I']], 
        'text': [f"LCL: {stats['LCL_I']:.2f}"]
    })).mark_text(align='right', dx=40, dy=-5, fontSize=11).encode(
        x='x:Q', y='y:Q', text='text:N'
    )
    
    return (line + cl + ucl + lcl + ucl_text + cl_text + lcl_text).properties(
        width=700,
        height=250,
        title='I-MR Chart of Test Result'
    )


def create_moving_range_chart(df, stats):
    """
    Create Moving Range chart (MR Chart) with smart Y-axis scaling
    
    Parameters:
    df: DataFrame with measurement data
    stats: Dictionary of statistics
    
    Returns:
    Altair chart object
    """
    chart_df = df.copy()
    chart_df['Observation'] = range(1, len(chart_df) + 1)
    chart_df['MR'] = abs(chart_df['Measure'].diff())
    
    # Remove first row (no MR value)
    chart_df = chart_df[chart_df['MR'].notna()]
    
    # Calculate smart Y range for MR chart
    mr_values = chart_df['MR'].values
    mr_max = np.max(mr_values)
    mr_range = stats['UCL_MR'] - stats['LCL_MR']
    
    # If max MR is much smaller than UCL, focus on data region
    if mr_max < stats['UCL_MR'] * 0.6:
        padding = mr_max * 0.2
        y_max = min(stats['UCL_MR'], mr_max + padding)
        margin = mr_range * 0.05
        if stats['UCL_MR'] - y_max < margin:
            y_max = stats['UCL_MR'] + margin
        y_min = stats['LCL_MR']
    else:
        # Standard range
        padding = mr_range * 0.05
        y_min = stats['LCL_MR']
        y_max = stats['UCL_MR'] + padding
    
    base = alt.Chart(chart_df).encode(
        x=alt.X('Observation:Q', 
                title='Observation', 
                scale=alt.Scale(domain=[0, len(df) + 1]))
    )
    
    # Moving Range line
    line = base.mark_line(color='blue', point=True).encode(
        y=alt.Y('MR:Q', 
                title='Moving Range', 
                scale=alt.Scale(domain=[y_min, y_max]))
    )
    
    # Center Line
    cl = alt.Chart(pd.DataFrame({'y': [stats['MR_bar']]})).mark_rule(
        color='green', strokeDash=[5, 5], size=2
    ).encode(y='y:Q')
    
    # UCL Line
    ucl = alt.Chart(pd.DataFrame({'y': [stats['UCL_MR']]})).mark_rule(
        color='red', size=2
    ).encode(y='y:Q')
    
    # LCL Line
    lcl = alt.Chart(pd.DataFrame({'y': [stats['LCL_MR']]})).mark_rule(
        color='red', size=2
    ).encode(y='y:Q')
    
    # Labels
    ucl_text = alt.Chart(pd.DataFrame({
        'x': [len(df)], 
        'y': [stats['UCL_MR']], 
        'text': [f"UCL: {stats['UCL_MR']:.2f}"]
    })).mark_text(align='right', dx=40, dy=-5, fontSize=11).encode(
        x='x:Q', y='y:Q', text='text:N'
    )
    
    cl_text = alt.Chart(pd.DataFrame({
        'x': [len(df)], 
        'y': [stats['MR_bar']], 
        'text': [f"MR̄: {stats['MR_bar']:.2f}"]
    })).mark_text(align='right', dx=40, dy=-5, fontSize=11).encode(
        x='x:Q', y='y:Q', text='text:N'
    )
    
    lcl_text = alt.Chart(pd.DataFrame({
        'x': [len(df)], 
        'y': [stats['LCL_MR']], 
        'text': [f"LCL: {stats['LCL_MR']:.2f}"]
    })).mark_text(align='right', dx=40, dy=-5, fontSize=11).encode(
        x='x:Q', y='y:Q', text='text:N'
    )
    
    return (line + cl + ucl + lcl + ucl_text + cl_text + lcl_text).properties(
        width=700,
        height=250
    )


def display_statistics(stats):
    """
    Display I-MR statistics in columns
    
    Parameters:
    stats: Dictionary of statistics
    """
    st.markdown("### I-MR Chart Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean (X̄)", f"{stats['X_bar']:.3f}")
        st.metric("MR̄", f"{stats['MR_bar']:.3f}")
    
    with col2:
        st.metric("UCL (I)", f"{stats['UCL_I']:.3f}")
        st.metric("UCL (MR)", f"{stats['UCL_MR']:.3f}")
    
    with col3:
        st.metric("LCL (I)", f"{stats['LCL_I']:.3f}")
        st.metric("LCL (MR)", f"{stats['LCL_MR']:.3f}")


def create_imr_chart(df):
    """
    Main function to create complete I-MR chart
    
    Parameters:
    df: DataFrame with 'Measure' column
    """
    # Calculate statistics
    stats = calculate_imr_statistics(df)
    
    # Create charts
    i_chart = create_individual_chart(df, stats)
    mr_chart = create_moving_range_chart(df, stats)
    
    # Combine charts
    combined_chart = alt.vconcat(i_chart, mr_chart).resolve_scale(x='shared')
    
    # Display
    st.altair_chart(combined_chart, use_container_width=True)
    
    # Display statistics
    display_statistics(stats)