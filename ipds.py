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
import base64
import io

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

# Enhanced CSS for Main App
def apply_wicket_css():
    css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@400;500&display=swap');
            
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
            
            .stSidebar .sidebar-content img {{
                filter: drop-shadow(0 0 10px {WICKET_THEME['accent']});
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
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
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []

# Authentication UI with HTML, CSS, and JS
def render_auth_ui():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NAMA IDPS Login</title>
        <style>
            :root {{
                --white: {WICKET_THEME['card_bg']};
                --gray: {WICKET_THEME['text']};
                --blue: {WICKET_THEME['button_bg']};
                --lightblue: {WICKET_THEME['accent_alt']};
                --button-radius: 0.7rem;
                --max-width: 758px;
                --max-height: 420px;
                font-size: 16px;
                font-family: 'Roboto Mono', monospace;
            }}

            body {{
                align-items: center;
                background-color: {WICKET_THEME['primary_bg']};
                background: url("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/airplane.jpg");
                background-attachment: fixed;
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
                display: grid;
                height: 100vh;
                place-items: center;
                overflow: hidden;
                margin: 0;
            }}

            .logo-container {{
                text-align: center;
                margin-bottom: 20px;
            }}

            .logo {{
                width: 150px;
                height: auto;
                filter: drop-shadow(0 0 10px {WICKET_THEME['accent']});
                animation: pulse 2s infinite;
            }}

            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}

            .form__title {{
                font-family: 'Orbitron', sans-serif;
                font-weight: 300;
                margin: 0;
                margin-bottom: 1.25rem;
                color: {WICKET_THEME['text_light']};
                text-shadow: 0 0 8px {WICKET_THEME['accent']};
            }}

            .link {{
                color: {WICKET_THEME['accent']};
                font-size: 0.9rem;
                margin: 1.5rem 0;
                text-decoration: none;
            }}

            .link:hover {{
                color: {WICKET_THEME['hover']};
            }}

            .container {{
                background-color: var(--white);
                backdrop-filter: blur(15px);
                border-radius: var(--button-radius);
                box-shadow: 0 0.9rem 1.7rem rgba(0, 0, 0, 0.25),
                    0 0.7rem 0.7rem rgba(0, 0, 0, 0.22);
                height: var(--max-height);
                max-width: var(--max-width);
                overflow: hidden;
                position: relative;
                width: 100%;
            }}

            .container__form {{
                height: 100%;
                position: absolute;
                top: 0;
                transition: all 0.6s ease-in-out;
            }}

            .container--signin {{
                left: 0;
                width: 50%;
                z-index: 2;
            }}

            .container.right-panel-active .container--signin {{
                transform: translateX(100%);
            }}

            .container--signup {{
                left: 0;
                opacity: 0;
                width: 50%;
                z-index: 1;
            }}

            .container.right-panel-active .container--signup {{
                animation: show 0.6s;
                opacity: 1;
                transform: translateX(100%);
                z-index: 5;
            }}

            .container__overlay {{
                height: 100%;
                left: 50%;
                overflow: hidden;
                position: absolute;
                top: 0;
                transition: transform 0.6s ease-in-out;
                width: 50%;
                z-index: 100;
            }}

            .container.right-panel-active .container__overlay {{
                transform: translateX(-100%);
            }}

            .overlay {{
                background-color: {WICKET_THEME['card_bg']};
                background: url("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/airplane.jpg");
                background-attachment: fixed;
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
                height: 100%;
                left: -100%;
                position: relative;
                transform: translateX(0);
                transition: transform 0.6s ease-in-out;
                width: 200%;
            }}

            .container.right-panel-active .overlay {{
                transform: translateX(50%);
            }}

            .overlay__panel {{
                align-items: center;
                display: flex;
                flex-direction: column;
                height: 100%;
                justify-content: center;
                position: absolute;
                text-align: center;
                top: 0;
                transform: translateX(0);
                transition: transform 0.6s ease-in-out;
                width: 50%;
            }}

            .overlay--left {{
                transform: translateX(-20%);
            }}

            .container.right-panel-active .overlay--left {{
                transform: translateX(0);
            }}

            .overlay--right {{
                right: 0;
                transform: translateX(0);
            }}

            .container.right-panel-active .overlay--right {{
                transform: translateX(20%);
            }}

            .btn {{
                background-color: var(--blue);
                background-image: linear-gradient(90deg, var(--blue) 0%, var(--lightblue) 74%);
                border-radius: 20px;
                border: 1px solid var(--blue);
                color: {WICKET_THEME['button_text']};
                cursor: pointer;
                font-family: 'Orbitron', sans-serif;
                font-size: 0.8rem;
                font-weight: bold;
                letter-spacing: 0.1rem;
                padding: 0.9rem 4rem;
                text-transform: uppercase;
                transition: transform 80ms ease-in;
            }}

            .form > .btn {{
                margin-top: 1.5rem;
            }}

            .btn:active {{
                transform: scale(0.95);
            }}

            .btn:focus {{
                outline: none;
            }}

            .form {{
                background-color: var(--white);
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                padding: 0 3rem;
                height: 100%;
                text-align: center;
            }}

            .input {{
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 8px;
                padding: 0.9rem;
                margin: 0.5rem 0;
                width: 100%;
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
            }}

            .input:focus {{
                border-color: {WICKET_THEME['accent']};
                box-shadow: 0 0 10px {WICKET_THEME['accent']};
                outline: none;
            }}

            @keyframes show {{
                0%, 49.99% {{
                    opacity: 0;
                    z-index: 1;
                }}
                50%, 100% {{
                    opacity: 1;
                    z-index: 5;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="logo-container">
            <img src="https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/nama_logo.jpg" alt="NAMA Logo" class="logo">
        </div>
        <div class="container right-panel-active">
            <div class="container__form container--signup">
                <form action="#" class="form" id="form1">
                    <h2 class="form__title">Sign Up</h2>
                    <input type="text" placeholder="Username" class="input" id="signup-username" />
                    <input type="email" placeholder="Email" class="input" id="signup-email" />
                    <input type="password" placeholder="Password" class="input" id="signup-password" />
                    <button class="btn" id="signup-btn">Sign Up</button>
                </form>
            </div>
            <div class="container__form container--signin">
                <form action="#" class="form" id="form2">
                    <h2 class="form__title">Sign In</h2>
                    <input type="text" placeholder="Username" class="input" id="signin-username" />
                    <input type="password" placeholder="Password" class="input" id="signin-password" />
                    <a href="#" class="link">Forgot your password?</a>
                    <button class="btn" id="signin-btn">Sign In</button>
                </form>
            </div>
            <div class="container__overlay">
                <div class="overlay">
                    <div class="overlay__panel overlay--left">
                        <button class="btn" id="signIn">Sign In</button>
                    </div>
                    <div class="overlay__panel overlay--right">
                        <button class="btn" id="signUp">Sign Up</button>
                    </div>
                </div>
            </div>
        </div>
        <script>
            const signInBtn = document.getElementById("signIn");
            const signUpBtn = document.getElementById("signUp");
            const firstForm = document.getElementById("form1");
            const secondForm = document.getElementById("form2");
            const container = document.querySelector(".container");

            signInBtn.addEventListener("click", () => {{
                container.classList.remove("right-panel-active");
            }});

            signUpBtn.addEventListener("click", () => {{
                container.classList.add("right-panel-active");
            }});

            firstForm.addEventListener("submit", (e) => {{
                e.preventDefault();
                alert("Sign-up functionality is disabled in this demo.");
            }});

            secondForm.addEventListener("submit", (e) => {{
                e.preventDefault();
                const username = document.getElementById("signin-username").value;
                const password = document.getElementById("signin-password").value;
                if (username === "nama" && password === "admin") {{
                    localStorage.setItem("authenticated", "true");
                    window.location.reload();
                }} else {{
                    alert("Invalid username or password");
                }}
            }});

            // Check if authenticated
            if (localStorage.getItem("authenticated") === "true") {{
                window.Streamlit = window.Streamlit || {{}};
                window.Streamlit.setAuthenticated = function() {{
                    Streamlit.sendMessage({{ type: 'authenticated' }});
                }};
                Streamlit.setAuthenticated();
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=600)
    if st.session_state.get('message') == 'authenticated':
        st.session_state.authenticated = True
        st.session_state.message = None
        st.rerun()

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
        fig.add_trace(go.Scattermapbox(
            lon=status_df['longitude'],
            lat=status_df['latitude'],
            mode='markers',
            marker=dict(
                size=12,
                color=WICKET_THEME['error'] if status == 'unidentified' else WICKET_THEME['success'],
                symbol='copter',
                opacity=0.8
            ),
            text=status_df['drone_id'],
            hovertemplate="%{text}<br>Altitude: %{customdata:.0f}m<br>Status: %{marker.color|status}<extra></extra>",
            customdata=status_df['altitude'],
            name=status.capitalize()
        ))
    fig.update_layout(
        mapbox=dict(
            style='streets-v12',
            center=dict(lat=9, lon=7),
            zoom=6,
            layers=[{'sourcetype': 'raster', 'source': ['mapbox://mapbox.terrain-rgb']}]
        ),
        showlegend=True,
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        title=dict(text="Drone Surveillance", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5),
        margin=dict(l=10, r=10, t=50, b=10)
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
    # Radar sweep effect (simulated as a cone)
    theta = np.linspace(0, 360, 100)
    r = np.ones(100) * 0.5  # Small radius for visual effect
    lon_sweep = 7 + r * np.cos(np.radians(theta))
    lat_sweep = 9 + r * np.sin(np.radians(theta))
    fig.add_trace(go.Scattermapbox(
        lon=lon_sweep,
        lat=lat_sweep,
        mode='lines',
        line=dict(color=WICKET_THEME['accent'], width=1),
        fill='toself',
        opacity=0.3,
        name='Radar Sweep'
    ))
    # Aircraft positions
    fig.add_trace(go.Scattermapbox(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='markers',
        marker=dict(
            size=12,
            color=WICKET_THEME['accent'],
            symbol='x',
            opacity=0.8
        ),
        text=df['target_id'],
        hovertemplate="%{text}<br>Altitude: %{customdata:.0f}ft<br>Velocity: %{customdata[1]:.0f}kts<extra></extra>",
        customdata=df[['altitude', 'velocity']].values,
        name='Radar Targets'
    ))
    fig.update_layout(
        mapbox=dict(
            style='streets-v12',
            center=dict(lat=9, lon=7),
            zoom=6,
            layers=[{'sourcetype': 'raster', 'source': ['mapbox://mapbox.terrain-rgb']}]
        ),
        showlegend=True,
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        title=dict(text="Radar Surveillance", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5),
        margin=dict(l=10, r=10, t=50, b=10)
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
            fig.add_trace(go.Scattermapbox(
                lon=anomaly_df['longitude'],
                lat=anomaly_df['latitude'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=WICKET_THEME['error'] if anomaly else WICKET_THEME['success'],
                    symbol='airfield',
                    opacity=0.8
                ),
                text=anomaly_df['icao24'],
                hovertemplate="%{text}<br>Altitude: %{customdata[0]:.0f}ft<br>Velocity: %{customdata[1]:.0f}kts<extra></extra>",
                customdata=anomaly_df[['altitude', 'velocity']].values,
                name='Anomaly' if anomaly else 'Normal'
            ))
    fig.update_layout(
        mapbox=dict(
            style='streets-v12',
            center=dict(lat=9, lon=7),
            zoom=6,
            layers=[{'sourcetype': 'raster', 'source': ['mapbox://mapbox.terrain-rgb']}]
        ),
        showlegend=True,
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        title=dict(text="ATC Monitoring", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5),
        margin=dict(l=10, r=10, t=50, b=10)
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

# Simulated Network Scan
def simulate_nmap_scan(target="192.168.1.1", scan_type="TCP SYN", port_range="1-1000"):
    common_ports = {
        21: ('ftp', 'tcp'), 22: ('ssh', 'tcp'), 23: ('telnet', 'tcp'), 80: ('http', 'tcp'),
        443: ('https', 'tcp'), 3306: ('mysql', 'tcp'), 3389: ('rdp', 'tcp')
    }
    start_port, end_port = map(int, port_range.split('-'))
    ports_to_scan = [p for p in common_ports.keys() if start_port <= p <= end_port]
    np.random.seed(42)
    scan_results = []
    for port in ports_to_scan:
        service, proto = common_ports[port]
        if scan_type == 'TCP SYN' and 'tcp' not in proto:
            continue
        if scan_type == 'UDP' and 'udp' not in proto:
            continue
        state = 'open' if np.random.random() > 0.5 else 'closed'
        scan_results.append({
            'port': port,
            'protocol': 'tcp' if scan_type != 'UDP' else 'udp',
            'state': state,
            'service': service
        })
    open_ports = len([r for r in scan_results if r['state'] == 'open'])
    st.session_state.compliance_metrics['open_ports'] = open_ports
    st.session_state.alert_log.append({
        'timestamp': datetime.now(),
        'type': 'Network Scan',
        'severity': 'medium',
        'details': f"Scanned {target}, found {open_ports} open ports"
    })
    logger.info(f"Simulated NMAP scan on {target}, found {open_ports} open ports")
    return scan_results

def display_network_scan():
    if not st.session_state.scan_results:
        st.warning("No scan data available. Click 'Simulate Scan' to generate data.")
        return
    df = pd.DataFrame(st.session_state.scan_results)
    st.dataframe(df[['port', 'protocol', 'state', 'service']])

# Generate Report
def generate_report():
    buffer = io.BytesIO()
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("NAMA IDPS Simulation Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    if st.session_state.alert_log:
        elements.append(Paragraph("Recent Alerts", styles['Heading2']))
        alert_data = [[str(a['timestamp']), a['type'], a['severity'], a['details']] for a in st.session_state.alert_log]
        alert_table = Table([['Timestamp', 'Type', 'Severity', 'Details']] + alert_data)
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(alert_table)
    
    if st.session_state.drone_results:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Drone Detection", styles['Heading2']))
        drone_data = [[str(d['timestamp']), d['drone_id'], f"{d['latitude']:.4f}", f"{d['longitude']:.4f}", f"{d['altitude']:.0f}", d['status']] for d in st.session_state.drone_results]
        drone_table = Table([['Timestamp', 'Drone ID', 'Latitude', 'Longitude', 'Altitude', 'Status']] + drone_data)
        drone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(drone_table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main Application
def main():
    apply_wicket_css()

    if not st.session_state.authenticated:
        render_auth_ui()
        return

    st.sidebar.image("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/nama_logo.jpg", use_column_width=True)
    page = st.sidebar.selectbox("Select Feature", ["Dashboard", "Network Scan", "Drone Detection", "Radar Surveillance", "ATC Monitoring", "Threat Intelligence", "Compliance Monitoring"])

    if page == "Dashboard":
        st.markdown('<div class="card"><h1>NAMA IDPS Simulation Dashboard</h1></div>', unsafe_allow_html=True)
        st.write("Select a feature from the sidebar to simulate and visualize its functionality.")
        if st.button("Generate All Simulations"):
            st.session_state.scan_results = simulate_nmap_scan()
            st.session_state.drone_results = simulate_drone_data()
            st.session_state.radar_data = simulate_radar_data()
            st.session_state.atc_results = simulate_atc_data()
            st.session_state.threats = simulate_threat_intelligence()
            st.session_state.compliance_metrics = simulate_compliance_metrics()
            st.success("Simulations generated!")
        if st.session_state.alert_log:
            st.subheader("Recent Alerts")
            st.dataframe(pd.DataFrame(st.session_state.alert_log[-5:]))
        if st.button("Download Report"):
            buffer = generate_report()
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="nama_idps_report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    elif page == "Network Scan":
        st.markdown('<div class="card"><h2>Network Scan Simulation</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Scan"):
            st.session_state.scan_results = simulate_nmap_scan()
        display_network_scan()

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
