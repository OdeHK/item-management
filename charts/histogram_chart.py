import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from scipy import stats


def calculate_bin_parameters(data, n_samples, LSL=None, USL=None):
    """
    Calculate optimal number of bins and bin width based on AIAG standards
    
    Parameters:
    data: Array of measurement values
    n_samples: Number of samples
    LSL: Lower Specification Limit
    USL: Upper Specification Limit
    
    Returns:
    dict: Dictionary with bin count and bin width
    """
    # Determine number of bins based on sample size (AIAG standard)
    if n_samples < 25:
        n_bins = 6
    elif 25 <= n_samples < 50:
        n_bins = 6
    elif 50 <= n_samples < 100:
        n_bins = 8
    elif 100 <= n_samples < 250:
        n_bins = 10
    elif 250 <= n_samples < 500:
        n_bins = 12
    elif 500 <= n_samples < 1000:
        n_bins = 14
    else:
        n_bins = 16
    
    # Calculate data range
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    
    # Use spec limits if provided, otherwise use data range with margin
    if LSL is not None and USL is not None:
        x_min = LSL
        x_max = USL
        plot_range = USL - LSL
    else:
        # Add 10% margin on each side for better visualization
        margin = data_range * 0.1
        x_min = data_min - margin
        x_max = data_max + margin
        plot_range = x_max - x_min
    
    # Calculate bin width based on plot range
    bin_width = plot_range / n_bins
    
    return {
        'n_bins': n_bins,
        'bin_width': bin_width,
        'data_min': data_min,
        'data_max': data_max,
        'data_range': data_range,
        'x_min': x_min,
        'x_max': x_max,
        'plot_range': plot_range
    }


def auto_calculate_spec_limits(data, stats):
    """
    Auto calculate specification limits based on data
    Using ±3σ or ±4σ from mean as spec limits
    
    Parameters:
    data: Array of measurement values
    stats: Dictionary with statistics
    
    Returns:
    dict: Dictionary with LSL and USL
    """
    mean = stats['mean']
    std = stats['std']
    
    # Use ±4σ for auto spec limits (covers ~99.99% of data)
    LSL = mean - 4 * std
    USL = mean + 4 * std
    
    # Make sure LSL and USL don't exceed actual data range too much
    data_min = stats['min']
    data_max = stats['max']
    
    # Adjust if needed
    if LSL < data_min - std:
        LSL = data_min - std
    if USL > data_max + std:
        USL = data_max + std
    
    return {
        'LSL': LSL,
        'USL': USL
    }
    """
    Calculate statistics needed for histogram and capability analysis
    
    Parameters:
    data: Array of measurement values
    
    Returns:
    dict: Dictionary with all statistics
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'min': np.min(data),
        'max': np.max(data)
    }


def create_normal_curve_data(mean, std, x_min, x_max, n_points=200):
    """
    Generate data points for normal distribution curve
    Using formula: y(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))
    
    Parameters:
    mean: Mean value (μ)
    std: Standard deviation (σ)
    x_min: Minimum x value
    x_max: Maximum x value
    n_points: Number of points to generate
    
    Returns:
    DataFrame with x and y values for normal curve
    """
    x = np.linspace(x_min, x_max, n_points)
    
    # Normal distribution PDF formula
    # y(x) = (1/(σ√(2π))) * e^(-(x-μ)²/(2σ²))
    coefficient = 1 / (std * np.sqrt(2 * np.pi))
    exponent = -((x - mean) ** 2) / (2 * std ** 2)
    y = coefficient * np.exp(exponent)
    
    return pd.DataFrame({'x': x, 'y': y})


def calculate_histogram_statistics(data):
    """
    Calculate statistics needed for histogram and capability analysis
    
    Parameters:
    data: Array of measurement values
    
    Returns:
    dict: Dictionary with all statistics
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'min': np.min(data),
        'max': np.max(data)
    }


def create_histogram_chart(data, bin_params, stats, LSL=None, USL=None, chart_width=650, chart_height=500):
    """
    Create histogram with normal distribution overlay
    
    Parameters:
    data: Array of measurement values
    bin_params: Dictionary with bin parameters
    stats: Dictionary with statistics
    LSL: Lower Specification Limit
    USL: Upper Specification Limit
    chart_width: Width of the chart in pixels
    chart_height: Height of the chart in pixels
    
    Returns:
    Altair chart object
    """
    # Create histogram data
    x_min = bin_params['x_min']
    x_max = bin_params['x_max']
    
    # Create bins
    bins = np.linspace(x_min, x_max, bin_params['n_bins'] + 1)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    
    # Create histogram dataframe
    hist_df = pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'count': hist,
        'probability': hist / len(data)  # Normalize to probability
    })
    hist_df['bin_center'] = (hist_df['bin_start'] + hist_df['bin_end']) / 2
    
    # Calculate dynamic bar width for visualization
    bar_width_ratio = 0.8  # 80% of bin width for visual appeal
    pixel_per_unit = chart_width / bin_params['plot_range']
    bar_width_pixels = bin_params['bin_width'] * pixel_per_unit * bar_width_ratio
    
    # Create histogram bars
    bars = alt.Chart(hist_df).mark_bar(
        color='#2ca02c',  # Teal/green color like in the image
        opacity=0.8,
        stroke='black',
        strokeWidth=0.5
    ).encode(
        x=alt.X('bin_center:Q', 
                title='Test Result',
                scale=alt.Scale(domain=[x_min, x_max]),
                axis=alt.Axis(format='.3f')),
        y=alt.Y('probability:Q', 
                title='Probability',
                scale=alt.Scale(domain=[0, max(hist_df['probability']) * 1.2])),
        size=alt.value(max(10, min(60, bar_width_pixels)))  # Clamp between 10-60 pixels
    )
    
    # Create normal distribution curve
    normal_df = create_normal_curve_data(
        stats['mean'], 
        stats['std'], 
        x_min, 
        x_max
    )
    
    # Scale the normal curve to match histogram height
    max_hist_prob = max(hist_df['probability']) if max(hist_df['probability']) > 0 else 1
    max_normal = max(normal_df['y'])
    scale_factor = max_hist_prob / max_normal * 0.95  # Scale to fit
    normal_df['y_scaled'] = normal_df['y'] * scale_factor
    
    # Create normal curve line
    normal_line = alt.Chart(normal_df).mark_line(
        color='red',
        strokeWidth=2.5
    ).encode(
        x=alt.X('x:Q'),
        y=alt.Y('y_scaled:Q')
    )
    
    # Add LSL and USL lines
    chart_layers = [bars, normal_line]
    
    if LSL is not None:
        lsl_line = alt.Chart(pd.DataFrame({'x': [LSL]})).mark_rule(
            color='red',
            strokeDash=[5, 5],
            strokeWidth=2
        ).encode(x='x:Q')
        
        lsl_text = alt.Chart(pd.DataFrame({
            'x': [LSL],
            'y': [max(hist_df['probability']) * 1.15],
            'text': ['LSL']
        })).mark_text(fontSize=12, dy=-5, color='red', fontWeight='bold').encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        chart_layers.extend([lsl_line, lsl_text])
    
    if USL is not None:
        usl_line = alt.Chart(pd.DataFrame({'x': [USL]})).mark_rule(
            color='red',
            strokeDash=[5, 5],
            strokeWidth=2
        ).encode(x='x:Q')
        
        usl_text = alt.Chart(pd.DataFrame({
            'x': [USL],
            'y': [max(hist_df['probability']) * 1.15],
            'text': ['USL']
        })).mark_text(fontSize=12, dy=-5, color='red', fontWeight='bold').encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        chart_layers.extend([usl_line, usl_text])
    
    # Combine all layers
    chart = alt.layer(*chart_layers).properties(
        width=chart_width,
        height=chart_height,
        title='Histogram'
    )
    
    return chart


def calculate_process_capability(stats_dict, LSL=None, USL=None):
    """
    Calculate process capability indices (Cp, Cpk, Pp, Ppk, PPM, Yield)
    
    Parameters:
    stats_dict: Dictionary with mean and std
    LSL: Lower Specification Limit
    USL: Upper Specification Limit
    
    Returns:
    dict: Dictionary with capability indices
    """
    if LSL is None or USL is None:
        return None
    
    mean = stats_dict['mean']
    std = stats_dict['std']
    
    # Cp = (USL - LSL) / (6σ)
    cp = (USL - LSL) / (6 * std)
    
    # Cpk = min((USL - μ)/(3σ), (μ - LSL)/(3σ))
    cpu = (USL - mean) / (3 * std)
    cpl = (mean - LSL) / (3 * std)
    cpk = min(cpu, cpl)
    
    # For Pp and Ppk, we use overall std (same as std for now)
    pp = (USL - LSL) / (6 * std)
    ppu = (USL - mean) / (3 * std)
    ppl = (mean - LSL) / (3 * std)
    ppk = min(ppu, ppl)
    
    # PPM = (1 - Φ(3*Cpk)) * 10^6 (approximation)
    # More accurate: use normal distribution from scipy.stats
    z_usl = (USL - mean) / std
    z_lsl = (LSL - mean) / std
    prob_within_spec = stats.norm.cdf(z_usl) - stats.norm.cdf(z_lsl)
    ppm = (1 - prob_within_spec) * 1e6
    
    # Yield = 1 - PPM/10^6
    yield_pct = prob_within_spec * 100
    
    return {
        'Cp': cp,
        'Cpk': cpk,
        'CPL': cpl,
        'CPU': cpu,
        'Pp': pp,
        'Ppk': ppk,
        'PPL': ppl,
        'PPU': ppu,
        'PPM': ppm,
        'Yield': yield_pct
    }


def display_capability_table(stats_dict, capability, LSL=None, USL=None, Target=None):
    """
    Display process capability table similar to Minitab format
    
    Parameters:
    stats_dict: Dictionary with statistics
    capability: Dictionary with capability indices
    LSL: Lower Specification Limit
    USL: Upper Specification Limit
    Target: Target value
    """
    st.markdown("### Process Capability")
    
    # Create the capability table
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("**Process Data**")
        st.text(f"LSL        {LSL if LSL is not None else 'N/A'}")
        st.text(f"Target     {Target if Target is not None else stats_dict['mean']:.3f}")
        st.text(f"USL        {USL if USL is not None else 'N/A'}")
        st.text(f"Mean       {stats_dict['mean']:.3f}")
        st.text(f"n          {stats_dict['n']}")
    
    with col2:
        st.markdown("**Within**")
        st.text(f"Std. Deviation   {stats_dict['std']:.3f}")
        if capability:
            st.text(f"Cp               {capability['Cp']:.2f}")
            st.text(f"CPL              {capability['CPL']:.2f}")
            st.text(f"CPU              {capability['CPU']:.2f}")
            st.text(f"Cpk              {capability['Cpk']:.2f}")
    
    with col3:
        st.markdown("**Overall**")
        st.text(f"Std. Deviation   {stats_dict['std']:.3f}")
        if capability:
            st.text(f"Pp               {capability['Pp']:.2f}")
            st.text(f"PPL              {capability['PPL']:.2f}")
            st.text(f"PPU              {capability['PPU']:.2f}")
            st.text(f"Ppk              {capability['Ppk']:.2f}")
            st.text(f"PPM              {capability['PPM']:.0f}")
            st.text(f"Yield (%)        {capability['Yield']:.2f}%")


def create_histogram(df, LSL=None, USL=None, Target=None, auto_spec_limits=True):
    """
    Main function to create histogram with normal curve and capability analysis
    
    Parameters:
    df: DataFrame with 'Measure' column
    LSL: Lower Specification Limit (optional)
    USL: Upper Specification Limit (optional)
    Target: Target value (optional)
    auto_spec_limits: If True and LSL/USL not provided, auto calculate them
    """
    data = df['Measure'].values
    
    # Calculate statistics
    stats_dict = calculate_histogram_statistics(data)
    
    # Auto calculate spec limits if not provided and auto_spec_limits is True
    if auto_spec_limits and (LSL is None or USL is None):
        spec_limits = auto_calculate_spec_limits(data, stats_dict)
        if LSL is None:
            LSL = spec_limits['LSL']
        if USL is None:
            USL = spec_limits['USL']
    
    # Calculate bin parameters with spec limits
    bin_params = calculate_bin_parameters(data, len(data), LSL, USL)
    
    # Calculate capability indices if spec limits provided
    capability = calculate_process_capability(stats_dict, LSL, USL)
    
    # Create 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Create and display histogram
        chart = create_histogram_chart(data, bin_params, stats_dict, LSL, USL, 
                                       chart_width=1000, chart_height=800)
        st.altair_chart(chart, use_container_width=False)
    
    with col2:
        # Import and display Normal Probability Plot
        from charts.probability_plot import create_probability_plot
        create_probability_plot(data, stats_dict['mean'], stats_dict['std'], 
                               chart_width=650, chart_height=500, show_test=True)
    
    # Display capability table (full width below charts)
    display_capability_table(stats_dict, capability, LSL, USL, Target)


def create_histogram_with_probability(df, LSL=None, USL=None, Target=None, auto_spec_limits=True):
    """
    Wrapper function to create histogram with probability plot
    This is the main function to call from app.py
    """
    create_histogram(df, LSL, USL, Target, auto_spec_limits)