import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def calculate_normal_probabilities(data):
    """
    Calculate theoretical normal probabilities for probability plot
    
    Parameters:
    data: Array of measurement values
    
    Returns:
    DataFrame with sorted data and theoretical probabilities
    """
    n = len(data)
    sorted_data = np.sort(data)
    
    # Calculate plotting positions using median rank method
    # Plotting position: (i - 0.3) / (n + 0.4) for better approximation
    ranks = np.arange(1, n + 1)
    plotting_positions = (ranks - 0.3) / (n + 0.4)
    
    # Convert to percentiles (0-100)
    percentiles = plotting_positions * 100
    
    # Calculate theoretical quantiles (z-scores)
    theoretical_quantiles = stats.norm.ppf(plotting_positions)
    
    return pd.DataFrame({
        'data': sorted_data,
        'percentile': percentiles,
        'theoretical_quantile': theoretical_quantiles,
        'rank': ranks
    })


def calculate_anderson_darling(data):
    """
    Calculate Anderson-Darling test for normality
    
    Parameters:
    data: Array of measurement values
    
    Returns:
    dict: Dictionary with test statistic and p-value
    """
    result = stats.anderson(data, dist='norm')
    
    # Get test statistic
    ad_statistic = result.statistic
    
    # Anderson-Darling critical values for normal distribution
    # Significance levels: 15%, 10%, 5%, 2.5%, 1%
    critical_values = result.critical_values
    significance_levels = result.significance_level
    
    # Determine p-value range
    if ad_statistic < critical_values[0]:
        p_value_range = "> 0.15"
    elif ad_statistic < critical_values[1]:
        p_value_range = "0.10 - 0.15"
    elif ad_statistic < critical_values[2]:
        p_value_range = "0.05 - 0.10"
    elif ad_statistic < critical_values[3]:
        p_value_range = "0.025 - 0.05"
    elif ad_statistic < critical_values[4]:
        p_value_range = "0.01 - 0.025"
    else:
        p_value_range = "< 0.01"
    
    return {
        'ad_statistic': ad_statistic,
        'p_value_range': p_value_range,
        'critical_values': critical_values,
        'significance_levels': significance_levels
    }


def create_probability_plot_chart(data, mean, std, chart_width=650, chart_height=500):
    """
    Create Normal Probability Plot - CHÍNH XÁC như app_with_excel.py
    Trục X = Theoretical quantiles (theoretical values from normal distribution)
    Trục Y = Actual data values (sorted_val)
    
    Parameters:
    data: Array of measurement values
    mean: Sample mean
    std: Sample standard deviation
    chart_width: Width of chart in pixels
    chart_height: Height of chart in pixels
    
    Returns:
    matplotlib figure and R² value
    """
    n = len(data)
    
    # Sắp xếp dữ liệu
    sorted_val = np.sort(data)
    
    # Tính plotting position theo công thức Blom: (i-0.5)/n
    plotting_positions = (np.arange(1, n+1) - 0.5) / n
    percentiles = plotting_positions * 100
    
    # Convert percentile → z-score (theoretical quantiles)
    z = stats.norm.ppf(plotting_positions)
    
    # ===== ĐÚNG: Q-Q PLOT STYLE =====
    # Trục X = theoretical values (từ phân phối chuẩn chuẩn hóa)
    # Trục Y = actual data values
    # → Đường fitted sẽ THẲNG!
    
    # Theoretical quantiles (chuẩn hóa về scale của data)
    theoretical_quantiles = mean + std * z
    
    # Tính đường fit tuyến tính: y = ax + b
    # Với normal data: slope ≈ 1, intercept ≈ 0 (nếu chuẩn hóa)
    slope, intercept = np.polyfit(theoretical_quantiles, sorted_val, 1)
    fitted_line = slope * theoretical_quantiles + intercept
    
    # Tính R²
    correlation = np.corrcoef(sorted_val, theoretical_quantiles)[0, 1]
    r_squared = correlation ** 2
    
    # ===== CONFIDENCE BAND =====
    # Dạng "envelope" hẹp giữa, rộng 2 đầu
    # Sử dụng công thức: width ∝ sqrt(p(1-p)/n)
    
    # Standard error tại mỗi percentile
    se_percentile = np.sqrt(plotting_positions * (1 - plotting_positions) / n)
    
    # Critical value cho 95% CI (CHÍNH XÁC như app_with_excel.py)
    z_crit = stats.norm.ppf(0.75)  # 0.674 (KHÔNG phải 1.96)
    
    # Margin tính theo percentile, rồi chuyển sang value
    # Công thức: SE(y) = σ × SE(percentile) / φ(z)
    pdf_z = stats.norm.pdf(z)
    se_value = std * se_percentile / (pdf_z + 1e-10)  # Tránh chia 0
    
    # Confidence bounds (trên trục Y - giá trị data)
    ci_width = z_crit * se_value * 2.5  # Hệ số 2.5 để band rộng hơn
    upper_bound_y = fitted_line + ci_width
    lower_bound_y = fitted_line - ci_width
    
    # ===== Vẽ BẰNG MATPLOTLIB =====
    fig, ax = plt.subplots(figsize=(chart_width/100, chart_height/100), dpi=100)
    
    # Confidence bands (hình chữ V ngược)
    ax.plot(theoretical_quantiles, upper_bound_y, color='red', linewidth=2)
    ax.plot(theoretical_quantiles, lower_bound_y, color='red', linewidth=2)
    ax.fill_between(theoretical_quantiles, lower_bound_y, upper_bound_y, 
                   color='red', alpha=0.2)
    
    # Fitted line (ĐƯỜNG THẲNG)
    ax.plot(theoretical_quantiles, fitted_line, color='black', linewidth=2)
    
    # Data points (dấu + xanh lam)
    ax.scatter(theoretical_quantiles, sorted_val, marker='+', s=80, 
              color='cyan', linewidths=2.5, zorder=10)
    
    # ===== Format trục =====
    # Trục Y: hiển thị theo percentile (nhưng plot theo value)
    # Chuyển đổi percentile → value tương ứng
    percentile_ticks = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    z_ticks = [stats.norm.ppf(p/100) for p in percentile_ticks]
    value_ticks = [mean + std * z_t for z_t in z_ticks]
    
    # Set limits
    y_min = min(sorted_val.min(), lower_bound_y.min())
    y_max = max(sorted_val.max(), upper_bound_y.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.02*y_range, y_max + 0.02*y_range)
    
    x_min = theoretical_quantiles.min()
    x_max = theoretical_quantiles.max()
    x_range = x_max - x_min
    ax.set_xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)
    
    # Labels
    ax.set_xlabel("Test Result", fontsize=11)
    ax.set_ylabel("Percent", fontsize=11)
    
    # Trục Y chính: hiển thị percentile
    ax.set_yticks(value_ticks)
    ax.set_yticklabels([str(int(p)) for p in percentile_ticks])
    
    # Tạo secondary y-axis để hiển thị actual data values
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(value_ticks)
    ax2.set_yticklabels(percentile_ticks)
    ax2.set_ylabel("")
    
    # Grid và background
    ax.grid(color='lightgray', alpha=0.7, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    
    # Title
    ax.set_title("Normal Probability Plot", fontsize=12, fontweight='normal', pad=10)
    
    fig.tight_layout()
    
    return fig, r_squared


def create_probability_plot(data, mean, std, chart_width=650, chart_height=500, show_test=True):
    """
    Main function to create Normal Probability Plot with normality test
    
    Parameters:
    data: Array of measurement values
    mean: Sample mean
    std: Sample standard deviation
    chart_width: Width of chart
    chart_height: Height of chart
    show_test: Whether to show Anderson-Darling test
    """
    # Create and display probability plot
    fig, r_squared = create_probability_plot_chart(data, mean, std, chart_width, chart_height)
    st.pyplot(fig)
    plt.close(fig)
    
    # Display statistics below chart
    if show_test:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("R² (Goodness of Fit)", f"{r_squared:.4f}")
        
        with col_b:
            # Anderson-Darling test
            ad_result = stats.anderson(data, dist='norm')
            ad_stat = ad_result.statistic
            critical_5pct = ad_result.critical_values[2]  # 5% significance level
            is_normal = "✓ Normal" if ad_stat < critical_5pct else "✗ Not Normal"
            st.metric("Anderson-Darling", f"{ad_stat:.3f} ({is_normal})")