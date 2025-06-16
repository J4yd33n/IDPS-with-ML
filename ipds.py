import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO, filename='nama_idps_sim.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced Cyberpunk Theme
WICKET_THEME = {
    "primary_bg": "#0A0F2D",
    "secondary_bg": "#1E2A44",
    "accent": "#00D4FF",
    "accent_alt": "#FF00FF",
    "text": "#E6E6FA",
    "text_light": "#FFFFFF",
    "card_bg": "rgba(30, 42, 68, 0.5)",
    "border": "#3B82F6",
    "button_bg": "#00D4FF",
    "button_text": "#0A0F2D",
    "hover": "#FF00FF",
    "error": "#FF4D4D",
    "success": "#00FF99"
}

# Enhanced CSS with Animations
def apply_wicket_css():
    css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@400;500&display=swap');
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
            
            .stApp {{
                background: linear-gradient(45deg, {WICKET_THEME['primary_bg']} 0%, {WICKET_THEME['secondary_bg']} 100%);
                background-size: 200% 200%;
                animation: gradientShift 15s ease-in-out infinite;
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
                overflow-x: hidden;
            }}
            
            @keyframes gradientShift {{
                0% {{ background-position: 0% 50%; }}
                50% {{ background-position: 100% 50%; }}
                100% {{ background-position: 0% 50%; }}
            }}
            
            .card {{
                background: {WICKET_THEME['card_bg']};
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                animation: fadeIn 1s ease-in-out;
            }}
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
            }}
            
            .stButton>button {{
                background: linear-gradient(45deg, {WICKET_THEME['button_bg']}, {WICKET_THEME['accent_alt']});
                color: {WICKET_THEME['button_text']};
                border-radius: 25px;
                padding: 12px 30px;
                border: none;
                font-family: 'Orbitron', sans-serif;
                font-weight: 700;
                letter-spacing: 1px;
                transition: all 0.3s ease;
                box-shadow: 0 0 10px {WICKET_THEME['button_bg']};
            }}
            .stButton>button:hover {{
                transform: scale(1.05);
                box-shadow: 0 0 20px {WICKET_THEME['hover']};
                background: linear-gradient(45deg, {WICKET_THEME['hover']}, {WICKET_THEME['accent_alt']});
            }}
            
            .plotly-graph-div {{
                background: {WICKET_THEME['card_bg']};
                border-radius: 12px;
                padding: 10px;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
            }}
            
            h1, h2, h3 {{
                font-family: 'Orbitron', sans-serif;
                color: {WICKET_THEME['text_light']};
                text-shadow: 0 0 8px {WICKET_THEME['accent']};
            }}
            
            .auth-container {{
                background: url('https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/images/aeroplane.jpg') no-repeat center center fixed;
                background-size: cover;
                position: relative;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}
            
            .auth-overlay {{
                position: absolute;
                inset: 0;
                background: linear-gradient(135deg, rgba(10, 15, 45, 0.85), rgba(30, 42, 68, 0.75));
                z-index: 1;
            }}
            
            .auth-card {{
                background: {WICKET_THEME['card_bg']};
                backdrop-filter: blur(15px);
                border: 2px solid {WICKET_THEME['border']};
                border-radius: 15px;
                padding: 2.5rem;
                max-width: 450px;
                width: 100%;
                box-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
                z-index: 2;
                animation: slideInAuth 0.8s ease-in-out forwards;
            }}
            
            @keyframes slideInAuth {{
                0% {{ transform: translateX(100vw); opacity: 0; }}
                100% {{ transform: translateX(0); opacity: 1; }}
            }}
            
            .auth-input {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 8px;
                padding: 12px;
                margin: 0.5rem 0;
                width: 100%;
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
            }}
            .auth-input:focus {{
                border-color: {WICKET_THEME['accent']};
                box-shadow: 0 0 12px {WICKET_THEME['accent']};
                outline: none;
            }}
            
            .auth-btn {{
                background: linear-gradient(45deg, {WICKET_THEME['button_bg']}, {WICKET_THEME['accent_alt']});
                color: {WICKET_THEME['button_text']};
                border-radius: 25px;
                padding: 12px 0;
                width: 100%;
                border: none;
                font-family: 'Orbitron', sans-serif;
                font-size: 0.9rem;
                font-weight: 700;
                cursor: pointer;
                margin-top: 1.5rem;
            }}
            .auth-btn:hover {{
                background: {WICKET_THEME['hover']};
                box-shadow: 0 0 25px {WICKET_THEME['hover']};
                transform: scale(1.05);
            }}
            
            .auth-link {{
                color: {WICKET_THEME['accent']};
                font-size: 0.9rem;
                margin-top: 1rem;
                text-decoration: none;
                font-family: 'Roboto Mono', monospace;
            }}
            .auth-link:hover {{
                color: {WICKET_THEME['hover']};
                text-shadow: 0 0 8px {WICKET_THEME['hover']};
            }}
            
            .radar {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(0, 212, 255, 0.3) 0%, transparent 70%);
                z-index: 2;
                animation: radarPulse 2s infinite;
            }}
            .radar::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 50%;
                width: 2px;
                height: 50%;
                background: {WICKET_THEME['accent']};
                transform-origin: bottom;
                animation: radarSweep 4s linear infinite;
            }}
            
            @keyframes radarPulse {{
                0% {{ transform: scale(1); opacity: 0.8; }}
                50% {{ transform: scale(1.2); opacity: 0.5; }}
                100% {{ transform: scale(1); opacity: 0.8; }}
            }}
            @keyframes radarSweep {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'drone_results' not in st.session_state:
    st.session_state.drone_results = []
if 'radar_data' not in st.session_state:
    st.session_state.radar_data = []
if 'atc_results' not in st.session_state:
    st.session_state.atc_results = []
if 'flight_conflicts' not in st.session_state:
    st.session_state.flight_conflicts = []
if 'threats' not in st.session_state:
    st.session_state.threats = []
if 'compliance_metrics' not in st.session_state:
    st.session_state.compliance_metrics = {'detection_rate': 0, 'open_ports': 0, 'alerts': 0}

# Simulated Drone Detection
def simulate_drone_data(num_drones=5):
    region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
    drones = []
    for i in range(num_drones):
        altitude = np.random.uniform(50, 1000)
        is_unauthorized = altitude < 400 and np.random.random() > 0.3
        drones.append({
            'timestamp': datetime.now(),
            'drone_id': f"DRN{i:03d}",
            'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
            'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
            'altitude': altitude,
            'status': 'unidentified' if is_unauthorized else 'authorized',
            'severity': 'high' if is_unauthorized else 'low'
        })
    if any(d['status'] == 'unidentified' for d in drones):
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Drone Intrusion',
            'severity': 'high',
            'details': f"Detected {sum(d['status'] == 'unidentified' for d in drones)} unauthorized drones"
        })
    logger.info(f"Simulated {len(drones)} drones, {sum(d['status'] == 'unidentified' for d in drones)} unauthorized")
    return drones

def display_drone_data():
    if not st.session_state.drone_results:
        st.warning("No drone data available. Click 'Simulate Drones' to generate data.")
        return
    df = pd.DataFrame(st.session_state.drone_results)
    fig = go.Figure()
    for status in df['status'].unique():
        status_df = df[df['status'] == status]
        fig.add_trace(go.Scattergeo(
            lon=status_df['longitude'],
            lat=status_df['latitude'],
            mode='markers',
            marker=dict(
                size=10,
                color=WICKET_THEME['error'] if status == 'unidentified' else WICKET_THEME['success'],
                symbol='triangle-up',
                line=dict(width=2, color=WICKET_THEME['text'])
            ),
            text=status_df['drone_id'],
            hoverinfo='text',
            name=status.capitalize()
        ))
    fig.update_layout(
        geo=dict(
            scope='africa',
            showland=True,
            landcolor=WICKET_THEME['secondary_bg'],
            showocean=True,
            oceancolor=WICKET_THEME['primary_bg'],
            showcountries=True,
            countrycolor=WICKET_THEME['border'],
            projection_type='mercator',
            center=dict(lat=9, lon=7),
            lataxis=dict(range=[4, 14]),
            lonaxis=dict(range=[2, 15])
        ),
        showlegend=True,
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        title=dict(text="Drone Surveillance", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['timestamp', 'drone_id', 'latitude', 'longitude', 'altitude', 'status', 'severity']])

# Simulated Radar Surveillance
def simulate_radar_data(num_targets=5):
    region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
    radar_data = []
    for i in range(num_targets):
        radar_data.append({
            'target_id': f"RAD{i:03d}",
            'timestamp': datetime.now(),
            'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
            'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
            'altitude': np.random.uniform(10000, 40000),
            'velocity': np.random.uniform(300, 600),
            'source': 'radar'
        })
    logger.info(f"Simulated {len(radar_data)} radar targets")
    return radar_data

def display_radar_data():
    if not st.session_state.radar_data:
        st.warning("No radar data available. Click 'Simulate Radar' to generate data.")
        return
    df = pd.DataFrame(st.session_state.radar_data)
    fig = go.Figure()
    # Radar sweep effect
    theta = np.linspace(0, 360, 100)
    r = np.ones(100) * 10
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        line=dict(color=WICKET_THEME['accent'], width=1),
        fill='toself',
        opacity=0.3,
        name='Radar Sweep'
    ))
    # Aircraft positions
    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='markers',
        marker=dict(
            size=12,
            color=WICKET_THEME['accent'],
            symbol='cross',
            line=dict(width=2, color=WICKET_THEME['text'])
        ),
        text=df['target_id'],
        hoverinfo='text',
        name='Radar Targets'
    ))
    fig.update_layout(
        geo=dict(
            scope='africa',
            showland=True,
            landcolor=WICKET_THEME['secondary_bg'],
            showocean=True,
            oceancolor=WICKET_THEME['primary_bg'],
            showcountries=True,
            countrycolor=WICKET_THEME['border'],
            projection_type='mercator',
            center=dict(lat=9, lon=7),
            lataxis=dict(range=[4, 14]),
            lonaxis=dict(range=[2, 15])
        ),
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 10]),
            angularaxis=dict(visible=False)
        ),
        showlegend=True,
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        title=dict(text="Radar Surveillance", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['timestamp', 'target_id', 'latitude', 'longitude', 'altitude', 'velocity']])

# Simulated ATC Monitoring
def simulate_atc_data(num_samples=10):
    region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
    data = []
    for i in range(num_samples):
        data.append({
            'timestamp': datetime.now(),
            'icao24': f"ICAO{i:04d}",
            'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
            'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
            'altitude': np.random.uniform(10000, 40000),
            'velocity': np.random.uniform(300, 600),
            'source': 'ads-b'
        })
    df = pd.DataFrame(data)
    # Anomaly detection
    features = ['latitude', 'longitude', 'altitude', 'velocity']
    X = df[features].fillna(0)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    df['anomaly'] = model.predict(X) == -1
    if df['anomaly'].any():
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'ATC Anomaly',
            'severity': 'high',
            'details': f"Detected {df['anomaly'].sum()} anomalies in ATC data"
        })
    # Collision risk detection
    conflicts = []
    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i+1:].iterrows():
            pos1 = (row1['latitude'], row1['longitude'])
            pos2 = (row2['latitude'], row2['longitude'])
            dist = geodesic(pos1, pos2).km
            if dist < 5:
                conflicts.append({
                    'icao24_1': row1['icao24'],
                    'icao24_2': row2['icao24'],
                    'distance_km': dist,
                    'severity': 'critical' if dist < 2 else 'high'
                })
    if conflicts:
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Flight Conflict',
            'severity': 'high',
            'details': f"Detected {len(conflicts)} collision risks"
        })
    st.session_state.flight_conflicts = conflicts
    logger.info(f"Simulated {len(data)} ATC records, {df['anomaly'].sum()} anomalies, {len(conflicts)} conflicts")
    return df.to_dict('records')

def display_atc_data():
    if not st.session_state.atc_results:
        st.warning("No ATC data available. Click 'Simulate ATC' to generate data.")
        return
    df = pd.DataFrame(st.session_state.atc_results)
    fig = go.Figure()
    for anomaly in [False, True]:
        anomaly_df = df[df['anomaly'] == anomaly]
        if not anomaly_df.empty:
            fig.add_trace(go.Scattergeo(
                lon=anomaly_df['longitude'],
                lat=anomaly_df['latitude'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=WICKET_THEME['error'] if anomaly else WICKET_THEME['success'],
                    symbol='circle',
                    line=dict(width=2, color=WICKET_THEME['text'])
                ),
                text=anomaly_df['icao24'],
                hoverinfo='text',
                name='Anomaly' if anomaly else 'Normal'
            ))
    fig.update_layout(
        geo=dict(
            scope='africa',
            showland=True,
            landcolor=WICKET_THEME['secondary_bg'],
            showocean=True,
            oceancolor=WICKET_THEME['primary_bg'],
            showcountries=True,
            countrycolor=WICKET_THEME['border'],
            projection_type='mercator',
            center=dict(lat=9, lon=7),
            lataxis=dict(range=[4, 14]),
            lonaxis=dict(range=[2, 15])
        ),
        showlegend=True,
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        title=dict(text="ATC Monitoring", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['timestamp', 'icao24', 'latitude', 'longitude', 'altitude', 'velocity', 'anomaly']])
    if st.session_state.flight_conflicts:
        st.subheader("Collision Risks")
        st.dataframe(pd.DataFrame(st.session_state.flight_conflicts))

# Simulated Threat Intelligence
def simulate_threat_intelligence(num_threats=5):
    threats = []
    for i in range(num_threats):
        threats.append({
            'timestamp': datetime.now(),
            'threat_id': f"THR{i:03d}",
            'description': f"Simulated threat {i+1}",
            'indicators': [f"192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}"],
            'severity': np.random.choice(['low', 'medium', 'high']),
            'source': 'simulated'
        })
    if any(t['severity'] in ['high', 'medium'] for t in threats):
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Threat Intelligence',
            'severity': 'high',
            'details': f"Detected {sum(t['severity'] in ['high', 'medium'] for t in threats)} notable threats"
        })
    logger.info(f"Simulated {len(threats)} threats")
    return threats

def display_threat_intelligence():
    if not st.session_state.threats:
        st.warning("No threat data available. Click 'Simulate Threats' to generate data.")
        return
    df = pd.DataFrame(st.session_state.threats)
    severity_counts = df['severity'].value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=severity_counts.index,
            y=severity_counts.values,
            marker_color=[WICKET_THEME['error'], WICKET_THEME['accent'], WICKET_THEME['success']],
            text=severity_counts.values,
            textposition='auto'
        )
    ])
    fig.update_layout(
        title="Threat Severity Distribution",
        xaxis_title="Severity",
        yaxis_title="Count",
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        font={'color': WICKET_THEME['text_light']}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['timestamp', 'threat_id', 'description', 'severity', 'source']])

# Simulated Compliance Monitoring
def simulate_compliance_metrics():
    metrics = {
        'detection_rate': np.random.uniform(70, 95),
        'open_ports': np.random.randint(0, 15),
        'alerts': len(st.session_state.alert_log)
    }
    if metrics['open_ports'] > 10 or metrics['alerts'] > 5:
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Compliance Alert',
            'severity': 'high',
            'details': f"High open ports ({metrics['open_ports']}) or alerts ({metrics['alerts']})"
        })
    logger.info(f"Simulated compliance metrics: {metrics}")
    return metrics

def display_compliance_metrics():
    if not st.session_state.compliance_metrics['detection_rate']:
        st.warning("No compliance data available. Click 'Simulate Compliance' to generate data.")
        return
    metrics = st.session_state.compliance_metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detection Rate", f"{metrics['detection_rate']:.1f}%")
    with col2:
        st.metric("Open Ports", metrics['open_ports'])
    with col3:
        st.metric("Active Alerts", metrics['alerts'])
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=metrics['detection_rate'],
        title={'text': "Intrusion Detection Rate"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': WICKET_THEME['accent']},
            'threshold': {
                'line': {'color': WICKET_THEME['error'], 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        font={'color': WICKET_THEME['text_light']}
    )
    st.plotly_chart(fig, use_container_width=True)

# Main Application
def main():
    apply_wicket_css()

    if not st.session_state.authenticated:
        st.markdown('<div class="auth-container"><div class="auth-overlay"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            st.markdown('<h2 style="text-align: center;">NAMA IDPS Simulation</h2>')
            username = st.text_input("Username", key="username", placeholder="Enter username", help="Default: nama")
            password = st.text_input("Password", type="password", key="password", placeholder="Enter password", help="Default: admin")
            if st.button("Sign In", key="signin_btn", help="Click to authenticate"):
                if username == "nama" and password == "admin":
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            st.markdown('<a href="#" class="auth-link">Forgot password?</a>', unsafe_allow_html=True)
            st.markdown('<div class="radar"></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.sidebar.title("NAMA IDPS Simulation")
    page = st.sidebar.selectbox("Select Feature", ["Dashboard", "Drone Detection", "Radar Surveillance", "ATC Monitoring", "Threat Intelligence", "Compliance Monitoring"])

    if page == "Dashboard":
        st.markdown('<div class="card"><h1>NAMA IDPS Simulation Dashboard</h1></div>', unsafe_allow_html=True)
        st.write("Select a feature from the sidebar to simulate and visualize its functionality.")
        if st.button("Generate All Simulations"):
            st.session_state.drone_results = simulate_drone_data()
            st.session_state.radar_data = simulate_radar_data()
            st.session_state.atc_results = simulate_atc_data()
            st.session_state.threats = simulate_threat_intelligence()
            st.session_state.compliance_metrics = simulate_compliance_metrics()
            st.success("Simulations generated!")
        if st.session_state.alert_log:
            st.subheader("Recent Alerts")
            st.dataframe(pd.DataFrame(st.session_state.alert_log[-5:]))

    elif page == "Drone Detection":
        st.markdown('<div class="card"><h2>Drone Detection Simulation</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Drones"):
            st.session_state.drone_results = simulate_drone_data()
        display_drone_data()

    elif page == "Radar Surveillance":
        st.markdown('<div class="card"><h2>Radar Surveillance Simulation</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Radar"):
            st.session_state.radar_data = simulate_radar_data()
        display_radar_data()

    elif page == "ATC Monitoring":
        st.markdown('<div class="card"><h2>ATC Monitoring Simulation</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate ATC"):
            st.session_state.atc_results = simulate_atc_data()
        display_atc_data()

    elif page == "Threat Intelligence":
        st.markdown('<div class="card"><h2>Threat Intelligence Simulation</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Threats"):
            st.session_state.threats = simulate_threat_intelligence()
        display_threat_intelligence()

    elif page == "Compliance Monitoring":
        st.markdown('<div class="card"><h2>Compliance Monitoring Simulation</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Compliance"):
            st.session_state.compliance_metrics = simulate_compliance_metrics()
        display_compliance_metrics()

if __name__ == "__main__":
    main()
