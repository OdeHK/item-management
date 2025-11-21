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


def create_individual_chart(df, stats):
    """
    Create Individual values chart (I Chart)
    
    Parameters:
    df: DataFrame with measurement data
    stats: Dictionary of statistics
    
    Returns:
    Altair chart object
    """
    chart_df = df.copy()
    chart_df['Observation'] = range(1, len(chart_df) + 1)
    chart_df['Individual'] = chart_df['Measure']
    
    base = alt.Chart(chart_df).encode(
        x=alt.X('Observation:Q', 
                title='Observation', 
                scale=alt.Scale(domain=[0, len(chart_df) + 1]))
    )
    
    # Individual values line
    line = base.mark_line(color='blue', point=True).encode(
        y=alt.Y('Individual:Q', 
                title='Individual values', 
                scale=alt.Scale(domain=[stats['LCL_I'] - 0.05, stats['UCL_I'] + 0.05]))
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
    Create Moving Range chart (MR Chart)
    
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
    
    base = alt.Chart(chart_df).encode(
        x=alt.X('Observation:Q', 
                title='Observation', 
                scale=alt.Scale(domain=[0, len(df) + 1]))
    )
    
    # Moving Range line
    line = base.mark_line(color='blue', point=True).encode(
        y=alt.Y('MR:Q', 
                title='Moving Range', 
                scale=alt.Scale(domain=[stats['LCL_MR'], stats['UCL_MR'] + 0.03]))
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
        'text': [f"M̄R: {stats['MR_bar']:.2f}"]
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