"""
Real-Time Transformer Load Monitoring System with Streamlit
Objective 1: Design a Real-Time Load Monitoring System

Run this with: streamlit run transformer_monitor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔌 Transformer Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.alert-high {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 1rem;
    margin: 0.5rem 0;
}
.alert-warning {
    background-color: #fff8e1;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    margin: 0.5rem 0;
}
.alert-normal {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = pd.DataFrame()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

# Functions
@st.cache_data
def load_historical_data(file_path):
    """Load and preprocess historical transformer data"""
    try:
        # Try to load from the specified path
        df = pd.read_csv(file_path)
        st.success(f"✅ Data loaded from {file_path}")
    except FileNotFoundError:
        st.warning("⚠️ File not found. Creating sample data for demonstration...")
        # Create comprehensive sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='H')
        np.random.seed(42)
        
        # More realistic load patterns
        hours = np.arange(len(dates)) % 24
        days = np.arange(len(dates)) // 24
        weekly_pattern = np.sin(days * 2 * np.pi / 7) * 5
        daily_pattern = np.sin(hours * 2 * np.pi / 24) * 8
        seasonal_pattern = np.sin(days * 2 * np.pi / 365) * 10
        
        df = pd.DataFrame({
            'DateTime': dates,
            'Load_kW': 55 + daily_pattern + weekly_pattern + seasonal_pattern + np.random.normal(0, 3, len(dates)),
            'Voltage': 229 + np.random.normal(0, 1.5, len(dates)) + np.sin(days * 2 * np.pi / 30) * 2,
            'Temperature': 29 + seasonal_pattern * 0.5 + np.random.normal(0, 2, len(dates))
        })
    
    # Preprocessing
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Feature engineering
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek
    df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
    df['Load_MA_24h'] = df['Load_kW'].rolling(window=24).mean()
    df['Voltage_Deviation'] = abs(df['Voltage'] - 230) / 230 * 100
    
    return df

def generate_real_time_point(historical_data):
    """Generate a single real-time data point"""
    if len(historical_data) == 0:
        return None
    
    # Get current time (simulate next hour)
    if len(st.session_state.real_time_data) > 0:
        last_time = st.session_state.real_time_data.index[-1]
        current_time = last_time + timedelta(hours=1)
    else:
        current_time = historical_data.index[-1] + timedelta(hours=1)
    
    # Get historical patterns for this hour
    hour = current_time.hour
    weekday = current_time.weekday()
    
    # Find similar historical data points
    similar_data = historical_data[
        (historical_data['Hour'] == hour) & 
        (historical_data['Weekday'] == weekday)
    ]
    
    if len(similar_data) > 0:
        load_base = similar_data['Load_kW'].mean()
        voltage_base = similar_data['Voltage'].mean()
        temp_base = similar_data['Temperature'].mean()
    else:
        load_base = historical_data['Load_kW'].mean()
        voltage_base = historical_data['Voltage'].mean()
        temp_base = historical_data['Temperature'].mean()
    
    # Add realistic variations
    load_new = max(20, load_base + np.random.normal(0, 4))
    voltage_new = max(200, min(250, voltage_base + np.random.normal(0, 1.5)))
    temp_new = max(15, min(50, temp_base + np.random.normal(0, 2.5)))
    
    # Occasionally introduce anomalies for demonstration
    if np.random.random() < 0.05:  # 5% chance of anomaly
        if np.random.random() < 0.5:
            load_new *= 1.3  # High load anomaly
        else:
            voltage_new += np.random.choice([-8, 8])  # Voltage anomaly
    
    new_point = pd.DataFrame({
        'Load_kW': [load_new],
        'Voltage': [voltage_new],
        'Temperature': [temp_new]
    }, index=[current_time])
    
    return new_point

def check_alerts(data_point, thresholds):
    """Check for alerts in the latest data point"""
    alerts = []
    timestamp = data_point.index[0]
    
    # Load alerts
    if data_point['Load_kW'].iloc[0] > thresholds['load_high']:
        alerts.append({
            'timestamp': timestamp,
            'type': 'HIGH_LOAD',
            'severity': 'HIGH',
            'message': f"⚠️ HIGH LOAD: {data_point['Load_kW'].iloc[0]:.1f} kW (Threshold: {thresholds['load_high']} kW)"
        })
    elif data_point['Load_kW'].iloc[0] < thresholds['load_low']:
        alerts.append({
            'timestamp': timestamp,
            'type': 'LOW_LOAD',
            'severity': 'MEDIUM',
            'message': f"⚠️ LOW LOAD: {data_point['Load_kW'].iloc[0]:.1f} kW (Threshold: {thresholds['load_low']} kW)"
        })
    
    # Voltage alerts
    if data_point['Voltage'].iloc[0] < thresholds['voltage_min']:
        alerts.append({
            'timestamp': timestamp,
            'type': 'LOW_VOLTAGE',
            'severity': 'HIGH',
            'message': f"🔻 LOW VOLTAGE: {data_point['Voltage'].iloc[0]:.1f} V (Min: {thresholds['voltage_min']} V)"
        })
    elif data_point['Voltage'].iloc[0] > thresholds['voltage_max']:
        alerts.append({
            'timestamp': timestamp,
            'type': 'HIGH_VOLTAGE',
            'severity': 'HIGH',
            'message': f"🔺 HIGH VOLTAGE: {data_point['Voltage'].iloc[0]:.1f} V (Max: {thresholds['voltage_max']} V)"
        })
    
    # Temperature alerts
    if data_point['Temperature'].iloc[0] > thresholds['temp_max']:
        alerts.append({
            'timestamp': timestamp,
            'type': 'HIGH_TEMPERATURE',
            'severity': 'MEDIUM',
            'message': f"🌡️ HIGH TEMP: {data_point['Temperature'].iloc[0]:.1f}°C (Max: {thresholds['temp_max']}°C)"
        })
    
    return alerts

def create_real_time_dashboard(historical_data, real_time_data):
    """Create the main dashboard"""
    
    # Combine data for visualization
    if len(real_time_data) > 0:
        # Show last 48 hours of historical + all real-time
        recent_historical = historical_data.tail(48)
        combined_data = pd.concat([recent_historical, real_time_data])
        latest_point = real_time_data.iloc[-1]
    else:
        combined_data = historical_data.tail(48)
        latest_point = historical_data.iloc[-1]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Load Monitoring (kW)', 'Voltage Monitoring (V)', 
                       'Temperature Monitoring (°C)', 'Load vs Voltage'),
        vertical_spacing=0.1
    )
    
    # Load plot
    if len(real_time_data) > 0:
        fig.add_trace(
            go.Scatter(x=recent_historical.index, y=recent_historical['Load_kW'],
                      name='Historical', line=dict(color='lightblue', width=1),
                      opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=real_time_data.index, y=real_time_data['Load_kW'],
                      name='Real-Time', line=dict(color='red', width=3),
                      mode='lines+markers'),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=combined_data.index, y=combined_data['Load_kW'],
                      name='Load', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Voltage plot
    fig.add_trace(
        go.Scatter(x=combined_data.index, y=combined_data['Voltage'],
                  name='Voltage', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_hline(y=230, line_dash="dash", line_color="black", 
                  annotation_text="Nominal", row=1, col=2)
    
    # Temperature plot
    fig.add_trace(
        go.Scatter(x=combined_data.index, y=combined_data['Temperature'],
                  name='Temperature', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Load vs Voltage scatter
    fig.add_trace(
        go.Scatter(x=combined_data['Voltage'], y=combined_data['Load_kW'],
                  mode='markers', name='Load vs Voltage',
                  marker=dict(color=combined_data.index.hour, colorscale='viridis')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, 
                     title_text="🔌 Real-Time Transformer Monitoring Dashboard")
    
    return fig, latest_point

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔌 Real-Time Transformer Monitoring System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Data loading section
    st.sidebar.subheader("📊 Data Source")
    data_path = st.sidebar.text_input("Data File Path:", 
                                    value="sample_data.csv"
    
    if st.sidebar.button("🔄 Load Data"):
        with st.spinner("Loading transformer data..."):
            st.session_state.historical_data = load_historical_data(data_path)
            st.session_state.data_loaded = True
            st.session_state.real_time_data = pd.DataFrame()  # Reset real-time data
            st.session_state.alerts = []  # Reset alerts
    
    # Alert thresholds
    st.sidebar.subheader("⚠️ Alert Thresholds")
    thresholds = {
        'load_high': st.sidebar.slider("High Load (kW)", 50, 100, 70),
        'load_low': st.sidebar.slider("Low Load (kW)", 10, 50, 30),
        'voltage_min': st.sidebar.slider("Min Voltage (V)", 220, 235, 225),
        'voltage_max': st.sidebar.slider("Max Voltage (V)", 230, 245, 235),
        'temp_max': st.sidebar.slider("Max Temperature (°C)", 30, 50, 40)
    }
    
    # Real-time monitoring controls
    st.sidebar.subheader("🔴 Real-Time Monitoring")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.monitoring_active = True
    with col2:
        if st.button("⏹️ Stop"):
            st.session_state.monitoring_active = False
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (5s)", value=True)
    
    if st.sidebar.button("➕ Add Data Point"):
        if st.session_state.data_loaded:
            new_point = generate_real_time_point(st.session_state.historical_data)
            if new_point is not None:
                st.session_state.real_time_data = pd.concat([st.session_state.real_time_data, new_point])
                
                # Check for alerts
                new_alerts = check_alerts(new_point, thresholds)
                st.session_state.alerts.extend(new_alerts)
                
                st.sidebar.success("✅ Data point added!")
        else:
            st.sidebar.error("❌ Please load data first!")
    
    # Clear data
    if st.sidebar.button("🗑️ Clear Real-Time Data"):
        st.session_state.real_time_data = pd.DataFrame()
        st.session_state.alerts = []
        st.sidebar.success("✅ Real-time data cleared!")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👆 Please load transformer data from the sidebar to begin monitoring.")
        return
    
    # Current status metrics
    st.subheader("📊 Current Status")
    
    if len(st.session_state.real_time_data) > 0:
        latest = st.session_state.real_time_data.iloc[-1]
        timestamp = st.session_state.real_time_data.index[-1]
    else:
        latest = st.session_state.historical_data.iloc[-1]
        timestamp = st.session_state.historical_data.index[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔌 Load",
            value=f"{latest['Load_kW']:.1f} kW",
            delta=f"{latest['Load_kW'] - st.session_state.historical_data['Load_kW'].mean():.1f}"
        )
    
    with col2:
        st.metric(
            label="⚡ Voltage", 
            value=f"{latest['Voltage']:.1f} V",
            delta=f"{latest['Voltage'] - 230:.1f}"
        )
    
    with col3:
        st.metric(
            label="🌡️ Temperature",
            value=f"{latest['Temperature']:.1f}°C",
            delta=f"{latest['Temperature'] - st.session_state.historical_data['Temperature'].mean():.1f}"
        )
    
    with col4:
        st.metric(
            label="📅 Last Update",
            value=timestamp.strftime("%H:%M:%S"),
            delta=f"{len(st.session_state.real_time_data)} new points"
        )
    
    # Alerts section
    if st.session_state.alerts:
        st.subheader("🚨 Active Alerts")
        
        # Show recent alerts
        recent_alerts = sorted(st.session_state.alerts, key=lambda x: x['timestamp'], reverse=True)[:5]
        
        for alert in recent_alerts:
            if alert['severity'] == 'HIGH':
                st.markdown(f'<div class="alert-high">{alert["message"]}<br><small>{alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</small></div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-warning">{alert["message"]}<br><small>{alert["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</small></div>', 
                           unsafe_allow_html=True)
    
    # Main dashboard
    st.subheader("📈 Real-Time Dashboard")
    
    if len(st.session_state.historical_data) > 0:
        dashboard_fig, latest_point = create_real_time_dashboard(
            st.session_state.historical_data, 
            st.session_state.real_time_data
        )
        st.plotly_chart(dashboard_fig, use_container_width=True)
    
    # Data tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recent Real-Time Data")
        if len(st.session_state.real_time_data) > 0:
            st.dataframe(st.session_state.real_time_data.tail(10).round(2), use_container_width=True)
        else:
            st.info("No real-time data yet. Click 'Add Data Point' to simulate new readings.")
    
    with col2:
        st.subheader("📊 Historical Summary")
        if len(st.session_state.historical_data) > 0:
            summary_stats = st.session_state.historical_data[['Load_kW', 'Voltage', 'Temperature']].describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
    
    # Auto-refresh mechanism
    if auto_refresh and st.session_state.monitoring_active:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()