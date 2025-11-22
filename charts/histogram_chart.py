import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from scipy import stats


def calculate_bin_parameters(data, n_samples, LSL=None, USL=None):
    """
    Calculate optimal number of bins and bin width based on AIAG standards
    Bins are based on ACTUAL DATA RANGE (not spec limits)
    Chart viewport expands to show LSL/USL if needed
    
    Parameters:
    data: Array of measurement values
    n_samples: Number of samples
    LSL: Lower Specification Limit (for chart viewport only)
    USL: Upper Specification Limit (for chart viewport only)
    
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
    
    # BINS cover ONLY data range (for histogram bars)
    # Use smaller margin to keep bins tight around data
    margin = data_range * 0.05 if data_range > 0 else 0.01
    bin_min = data_min - margin
    bin_max = data_max + margin
    bin_range = bin_max - bin_min
    
    # Calculate bin width based on data range
    bin_width = bin_range / n_bins
    
    # CHART VIEWPORT expands to show LSL/USL if needed
    chart_min = bin_min
    chart_max = bin_max
    
    if LSL is not None and LSL < chart_min:
        chart_min = LSL - (data_range * 0.05)
    
    if USL is not None and USL > chart_max:
        chart_max = USL + (data_range * 0.05)
    
    return {
        'n_bins': n_bins,
        'bin_width': bin_width,
        'data_min': data_min,
        'data_max': data_max,
        'data_range': data_range,
        'bin_min': bin_min,      # For creating bins
        'bin_max': bin_max,      # For creating bins
        'chart_min': chart_min,  # For chart x-axis
        'chart_max': chart_max   # For chart x-axis
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


def create_histogram_chart(data, bin_params, stats, LSL=None, USL=None, Target=None):
    """
    Create histogram with normal distribution overlay (responsive)
    Bins cover data range only, chart viewport expands for LSL/USL
    
    Parameters:
    data: Array of measurement values
    bin_params: Dictionary with bin parameters
    stats: Dictionary with statistics
    LSL: Lower Specification Limit (reference line)
    USL: Upper Specification Limit (reference line)
    Target: Target value (reference line)
    
    Returns:
    Altair chart object
    """
    # Bins cover ONLY data range (tight fit)
    bin_min = bin_params['bin_min']
    bin_max = bin_params['bin_max']
    
    # Chart viewport (may be wider to show LSL/USL)
    chart_min = bin_params['chart_min']
    chart_max = bin_params['chart_max']
    data_range = bin_params['data_range']
    
    # Create bins based on DATA RANGE ONLY
    bins = np.linspace(bin_min, bin_max, bin_params['n_bins'] + 1)
    
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
    
    # Auto-detect appropriate format based on data range
    if data_range < 1:
        # Very small values (like 0.2-0.5) - use 3 decimal places
        format_str = '.3f'
        tick_count = 8  # More ticks for small ranges
    elif data_range < 10:
        # Small values (like 1-10) - use 2 decimal places
        format_str = '.2f'
        tick_count = 10
    elif data_range < 100:
        # Medium values (like 10-100) - use 1 decimal place
        format_str = '.1f'
        tick_count = 10
    else:
        # Large values (like 1614-1615) - use 1 decimal place
        format_str = '.1f'
        tick_count = 10
    
    # Create histogram bars (responsive - no fixed size)
    bars = alt.Chart(hist_df).mark_bar(
        color='#2ca02c',  # Green color
        opacity=0.8,
        stroke='black',
        strokeWidth=2,
        size=50
    ).encode(
        x=alt.X('bin_center:Q', 
                title='Test Result',
                scale=alt.Scale(domain=[chart_min, chart_max]),  # Wider viewport
                axis=alt.Axis(
                    format=format_str, 
                    labelAngle=-45,
                    tickCount=tick_count,
                    grid=True
                )),
        y=alt.Y('probability:Q', 
                title='Probability',
                scale=alt.Scale(domain=[0, max(hist_df['probability']) * 1.2 if len(hist_df) > 0 else 1]),
                axis=alt.Axis(grid=True))
    )
    
    # Create normal distribution curve (over DATA range, not chart viewport)
    normal_df = create_normal_curve_data(
        stats['mean'], 
        stats['std'], 
        bin_min,  # Normal curve covers data range
        bin_max
    )
    
    # Scale the normal curve to match histogram height
    max_hist_prob = max(hist_df['probability']) if len(hist_df) > 0 and max(hist_df['probability']) > 0 else 1
    max_normal = max(normal_df['y']) if len(normal_df) > 0 else 1
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
    
    # Add reference lines: LSL, USL, Target
    # NOTE: These may be outside the visible chart area
    chart_layers = [bars, normal_line]
    
    max_y = max(hist_df['probability']) if len(hist_df) > 0 and max(hist_df['probability']) > 0 else 1
    max_y = max_y * 1.15
    
    # Determine format for reference line labels
    if data_range < 1:
        ref_format = '.3f'
    elif data_range < 10:
        ref_format = '.2f'
    else:
        ref_format = '.1f'
    
    # Add LSL line (even if outside viewport)
    if LSL is not None:
        # Check if LSL is visible in chart
        lsl_visible = (LSL >= chart_min and LSL <= chart_max)
        
        lsl_line = alt.Chart(pd.DataFrame({'x': [LSL]})).mark_rule(
            color='red',
            strokeDash=[5, 5],
            strokeWidth=2,
            opacity=1.0 if lsl_visible else 0.5  # Dim if outside
        ).encode(x='x:Q')
        
        # Position label based on visibility
        if lsl_visible:
            label_x = LSL
        elif LSL < chart_min:
            label_x = chart_min + (chart_max - chart_min) * 0.05  # 5% from left edge
        else:
            label_x = chart_max - (chart_max - chart_min) * 0.05  # 5% from right edge
        
        lsl_label = f"LSL\n{LSL:{ref_format}}" if data_range < 100 else f"LSL"
        
        lsl_text = alt.Chart(pd.DataFrame({
            'x': [label_x],
            'y': [max_y],
            'text': [lsl_label]
        })).mark_text(
            fontSize=11, 
            dy=-10, 
            color='red', 
            fontWeight='bold',
            align='center'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        chart_layers.extend([lsl_line, lsl_text])
    
    # Add USL line (even if outside viewport)
    if USL is not None:
        # Check if USL is visible in chart
        usl_visible = (USL >= chart_min and USL <= chart_max)
        
        usl_line = alt.Chart(pd.DataFrame({'x': [USL]})).mark_rule(
            color='red',
            strokeDash=[5, 5],
            strokeWidth=2,
            opacity=1.0 if usl_visible else 0.5  # Dim if outside
        ).encode(x='x:Q')
        
        # Position label based on visibility
        if usl_visible:
            label_x = USL
        elif USL < chart_min:
            label_x = chart_min + (chart_max - chart_min) * 0.05
        else:
            label_x = chart_max - (chart_max - chart_min) * 0.05
        
        usl_label = f"USL\n{USL:{ref_format}}" if data_range < 100 else f"USL"
        
        usl_text = alt.Chart(pd.DataFrame({
            'x': [label_x],
            'y': [max_y],
            'text': [usl_label]
        })).mark_text(
            fontSize=11, 
            dy=-10, 
            color='red', 
            fontWeight='bold',
            align='center'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        chart_layers.extend([usl_line, usl_text])
    
    # Add Target line (even if outside viewport)
    if Target is not None:
        # Check if Target is visible in chart
        target_visible = (Target >= chart_min and Target <= chart_max)
        
        target_line = alt.Chart(pd.DataFrame({'x': [Target]})).mark_rule(
            color='green',
            strokeDash=[3, 3],
            strokeWidth=2,
            opacity=1.0 if target_visible else 0.5  # Dim if outside
        ).encode(x='x:Q')
        
        # Position label based on visibility
        if target_visible:
            label_x = Target
        elif Target < chart_min:
            label_x = chart_min + (chart_max - chart_min) * 0.05
        else:
            label_x = chart_max - (chart_max - chart_min) * 0.05
        
        target_label = f"Target\n{Target:{ref_format}}" if data_range < 100 else f"Target"
        
        target_text = alt.Chart(pd.DataFrame({
            'x': [label_x],
            'y': [max_y],
            'text': [target_label]
        })).mark_text(
            fontSize=11, 
            dy=-10, 
            color='green', 
            fontWeight='bold',
            align='center'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )
        chart_layers.extend([target_line, target_text])
    
    # Combine all layers (responsive - container width)
    chart = alt.layer(*chart_layers).properties(
        height=500,  # Increased height from 450 to 500
        title='Histogram'
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    ).configure_view(
        strokeWidth=0  # Remove border
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
    if LSL is None and USL is None:
        return None
    
    mean = stats_dict['mean']
    std = stats_dict['std']
    
    # Initialize all values
    cp = None
    cpu = None
    cpl = None
    cpk = None
    pp = None
    ppu = None
    ppl = None
    ppk = None
    ppm = 0
    yield_pct = 100
    
    # Calculate Cp only if both limits exist
    if LSL is not None and USL is not None:
        cp = (USL - LSL) / (6 * std)
        pp = (USL - LSL) / (6 * std)
    
    # Calculate CPU if USL exists
    if USL is not None:
        cpu = (USL - mean) / (3 * std)
        ppu = (USL - mean) / (3 * std)
    
    # Calculate CPL if LSL exists
    if LSL is not None:
        cpl = (mean - LSL) / (3 * std)
        ppl = (mean - LSL) / (3 * std)
    
    # Calculate Cpk and Ppk
    if cpu is not None and cpl is not None:
        cpk = min(cpu, cpl)
        ppk = min(ppu, ppl)
    elif cpu is not None:
        cpk = cpu
        ppk = ppu
    elif cpl is not None:
        cpk = cpl
        ppk = ppl
    
    # Calculate PPM and Yield
    if LSL is not None and USL is not None:
        z_usl = (USL - mean) / std
        z_lsl = (LSL - mean) / std
        prob_within_spec = stats.norm.cdf(z_usl) - stats.norm.cdf(z_lsl)
    elif USL is not None:
        z_usl = (USL - mean) / std
        prob_within_spec = stats.norm.cdf(z_usl)
    elif LSL is not None:
        z_lsl = (LSL - mean) / std
        prob_within_spec = 1 - stats.norm.cdf(z_lsl)
    else:
        prob_within_spec = 1.0
    
    ppm = (1 - prob_within_spec) * 1e6
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
    
    # Create the capability table with 2-column alignment
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("**Process Data**")
        
        # Split into 2 sub-columns for parameter and value
        subcol1, subcol2 = st.columns([1.5, 1])
        
        with subcol1:
            st.markdown("LSL")
            st.markdown("Target")
            st.markdown("USL")
            st.markdown("Mean")
            st.markdown("n")
        
        with subcol2:
            st.markdown(f"{LSL:.3f}" if LSL is not None else "N/A")
            st.markdown(f"{Target:.3f}" if Target is not None else f"{stats_dict['mean']:.3f}")
            st.markdown(f"{USL:.3f}" if USL is not None else "N/A")
            st.markdown(f"{stats_dict['mean']:.3f}")
            st.markdown(f"{stats_dict['n']}")
    
    with col2:
        st.markdown("**Within**")
        
        # Split into 2 sub-columns for parameter and value
        subcol1, subcol2 = st.columns([1.5, 1])
        
        with subcol1:
            st.markdown("Std. Deviation")
            st.markdown("Cp")
            st.markdown("CPL")
            st.markdown("CPU")
            st.markdown("Cpk")
        
        with subcol2:
            st.markdown(f"{stats_dict['std']:.3f}")
            
            if capability:
                st.markdown(f"{capability['Cp']:.2f}" if capability['Cp'] is not None else "N/A")
                st.markdown(f"{capability['CPL']:.2f}" if capability['CPL'] is not None else "N/A")
                st.markdown(f"{capability['CPU']:.2f}" if capability['CPU'] is not None else "N/A")
                st.markdown(f"{capability['Cpk']:.2f}" if capability['Cpk'] is not None else "N/A")
            else:
                st.markdown("N/A")
                st.markdown("N/A")
                st.markdown("N/A")
                st.markdown("N/A")
    
    with col3:
        st.markdown("**Overall**")
        
        # Split into 2 sub-columns for parameter and value
        subcol1, subcol2 = st.columns([1.5, 1])
        
        with subcol1:
            st.markdown("Std. Deviation")
            st.markdown("Pp")
            st.markdown("PPL")
            st.markdown("PPU")
            st.markdown("Ppk")
            st.markdown("PPM")
            st.markdown("Yield (%)")
        
        with subcol2:
            st.markdown(f"{stats_dict['std']:.3f}")
            
            if capability:
                st.markdown(f"{capability['Pp']:.2f}" if capability['Pp'] is not None else "N/A")
                st.markdown(f"{capability['PPL']:.2f}" if capability['PPL'] is not None else "N/A")
                st.markdown(f"{capability['PPU']:.2f}" if capability['PPU'] is not None else "N/A")
                st.markdown(f"{capability['Ppk']:.2f}" if capability['Ppk'] is not None else "N/A")
                st.markdown(f"{capability['PPM']:.0f}" if capability['PPM'] is not None else "N/A")
                st.markdown(f"{capability['Yield']:.2f}" if capability['Yield'] is not None else "N/A")
            else:
                st.markdown("N/A")
                st.markdown("N/A")
                st.markdown("N/A")
                st.markdown("N/A")
                st.markdown("N/A")
                st.markdown("N/A")


def create_histogram(df, LSL=None, USL=None, Target=None, auto_spec_limits=False):
    """
    Main function to create histogram with normal curve and capability analysis
    
    Parameters:
    df: DataFrame with 'Measure' column
    LSL: Lower Specification Limit (optional)
    USL: Upper Specification Limit (optional)
    Target: Target value (optional)
    auto_spec_limits: Not used anymore (kept for compatibility)
    """
    data = df['Measure'].values
    
    # Calculate statistics
    stats_dict = calculate_histogram_statistics(data)
    
    # Calculate bin parameters based on DATA (not spec limits)
    bin_params = calculate_bin_parameters(data, len(data), LSL, USL)
    
    # Calculate capability indices if spec limits provided
    capability = calculate_process_capability(stats_dict, LSL, USL)
    
    # Create 2-column layout (responsive)
    col1, col2 = st.columns(2)
    
    with col1:
        # Create and display histogram with Target line
        chart = create_histogram_chart(data, bin_params, stats_dict, LSL, USL, Target)
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        # Import and display Normal Probability Plot (responsive)
        from charts.probability_plot import create_probability_plot
        create_probability_plot(data, stats_dict['mean'], stats_dict['std'], 
                               chart_width=6.5, chart_height=5, show_test=True)
    
    # Display capability table (full width below charts)
    display_capability_table(stats_dict, capability, LSL, USL, Target)


def create_histogram_with_probability(df, LSL=None, USL=None, Target=None, auto_spec_limits=False):
    """
    Wrapper function to create histogram with probability plot
    This is the main function to call from app.py
    """
    create_histogram(df, LSL, USL, Target, auto_spec_limits)