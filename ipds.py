import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
import base64
import io
import sys
import sqlite3
import bcrypt
import re

# Configure logging
logging.basicConfig(level=logging.INFO, filename='nama_idps_sim.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Log to stderr for Streamlit Cloud

# Log Python and package versions
logger.info(f"Python version: {sys.version}")
try:
    import streamlit
    import pandas
    import numpy
    import plotly
    import geopy
    import sklearn
    import reportlab
    import bcrypt
    logger.info(f"Streamlit: {streamlit.__version__}, Pandas: {pandas.__version__}, "
                f"Numpy: {numpy.__version__}, Plotly: {plotly.__version__}, "
                f"Geopy: {geopy.__version__}, Scikit-learn: {sklearn.__version__}, "
                f"Reportlab: {reportlab.__version__}, Bcrypt: {bcrypt.__version__}")
except ImportError as e:
    logger.error(f"Dependency import failed: {str(e)}")

# Initialize SQLite database
def init_db():
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (username TEXT PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
            conn.commit()
            logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {str(e)}")
        st.error("Database error. Please try again later.")

# Validate email format
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

# Validate password strength
def is_valid_password(password):
    return len(password) >= 8 and any(c.isupper() for c in password) and any(c.isdigit() for c in password)

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
                background: url('https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/airplane.jpg');
                background-size: cover;
                background-position: center;
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
                overflow-x: hidden;
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
            
            .auth-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
            }}
            
            .form-container {{
                background: {WICKET_THEME['card_bg']};
                backdrop-filter: blur(15px);
                border-radius: 10px;
                padding: 30px;
                width: 100%;
                max-width: 400px;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
                position: relative;
                transition: transform 0.6s ease-in-out;
            }}
            
            .form-container.sign-up-active {{
                transform: translateX(100%);
            }}
            
            .stTextInput input, .stTextInput input:focus {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 8px;
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
            }}
            
            .stTextInput input:focus {{
                border-color: {WICKET_THEME['accent']};
                box-shadow: 0 0 10px {WICKET_THEME['accent']};
            }}
            
            .logo {{
                display: block;
                margin: 0 auto 20px;
                width: 150px;
            }}
            
            .forgot-password {{
                color: {WICKET_THEME['accent']};
                font-size: 0.9rem;
                text-decoration: none;
                display: block;
                margin: 10px 0;
            }}
            
            .forgot-password:hover {{
                color: {WICKET_THEME['hover']};
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
if 'panel_state' not in st.session_state:
    st.session_state.panel_state = 'sign_in'

# Initialize database
init_db()

# Authentication UI with Native Streamlit
def render_auth_ui():
    st.markdown(
        '<div class="auth-container">'
        f'<img src="https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/nama_logo.jpg" class="logo">'
        f'<div class="form-container {"sign-up-active" if st.session_state.panel_state == "sign_up" else ""}">',
        unsafe_allow_html=True
    )

    if st.session_state.panel_state == 'sign_in':
        with st.form(key='sign_in_form'):
            st.markdown('<h2 style="text-align: center;">Sign In</h2>', unsafe_allow_html=True)
            username = st.text_input('Username', placeholder='Username')
            password = st.text_input('Password', type='password', placeholder='Password')
            st.markdown('<a href="#" class="forgot-password">Forgot your password?</a>', unsafe_allow_html=True)
            submit = st.form_submit_button('Sign In')
            if submit:
                try:
                    with sqlite3.connect('users.db') as conn:
                        c = conn.cursor()
                        c.execute('SELECT password FROM users WHERE username = ?', (username,))
                        result = c.fetchone()
                        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
                            st.session_state.authenticated = True
                            logger.info(f"Authenticated: username={username}")
                            st.rerun()
                        else:
                            st.error('Invalid username or password')
                            logger.warning(f"Failed login attempt: username={username}")
                except sqlite3.Error as e:
                    st.error('Database error during login')
                    logger.error(f"Login database error: {str(e)}")
        st.button('Switch to Sign Up', on_click=lambda: st.session_state.update(panel_state='sign_up'))
    else:
        with st.form(key='sign_up_form'):
            st.markdown('<h2 style="text-align: center;">Sign Up</h2>', unsafe_allow_html=True)
            username = st.text_input('Username', placeholder='Username', key='signup_username')
            email = st.text_input('Email', placeholder='Email')
            password = st.text_input('Password', type='password', placeholder='Password', key='signup_password')
            submit = st.form_submit_button('Sign Up')
            if submit:
                if not username or not email or not password:
                    st.error('All fields are required')
                elif not is_valid_email(email):
                    st.error('Invalid email format')
                elif not is_valid_password(password):
                    st.error('Password must be at least 8 characters, include an uppercase letter and a number')
                else:
                    try:
                        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                        with sqlite3.connect('users.db') as conn:
                            c = conn.cursor()
                            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                                      (username, email, hashed_pw.decode('utf-8')))
                            conn.commit()
                            st.success('Account created! Please sign in.')
                            st.session_state.panel_state = 'sign_in'
                            logger.info(f"New user registered: username={username}")
                            st.rerun()
                    except sqlite3.IntegrityError:
                        st.error('Username or email already exists')
                        logger.warning(f"Sign-up failed: username={username}, email={email} already exists")
                    except sqlite3.Error as e:
                        st.error('Database error during sign-up')
                        logger.error(f"Sign-up database error: {str(e)}")
        st.button('Switch to Sign In', on_click=lambda: st.session_state.update(panel_state='sign_in'))

    st.markdown('</div></div>', unsafe_allow_html=True)

# Simulated Drone Detection
def simulate_drone_data(num_drones=30):
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
    try:
        fig = go.Figure()
        for status in df['status'].unique():
            status_df = df[df['status'] == status]
            fig.add_trace(go.Scattermapbox(
                lon=status_df['longitude'],
                lat=status_df['latitude'],
                mode='markers',
                marker=dict(
                    size=10,  # Smaller size for 30 drones
                    color=WICKET_THEME['error'] if status == 'unidentified' else WICKET_THEME['success'],
                    symbol='copter',
                    opacity=0.8,
                    allowoverlap=True
                ),
                text=status_df['drone_id'],
                hovertemplate="%{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>Altitude: %{customdata:.0f}m<br>Status: %{marker.color|status}<extra></extra>",
                customdata=status_df['altitude'],
                name=status.capitalize()
            ))
        mapbox_token = st.secrets.get('MAPBOX_TOKEN', 'pk.eyJ1IjoiZ3JvazMiLCJhIjoiY2x6aG5sOHVrMDM3NjJrbzF0M3A0eTRsZyJ9._ZJqGa-3kT-Zv2Cto0L_2Q')
        fig.update_layout(
            mapbox=dict(
                style='streets-v12',  # Fallback to streets
                center=dict(lat=9, lon=7),
                zoom=7,
                accesstoken=mapbox_token
            ),
            showlegend=True,
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            title=dict(text="Drone Surveillance", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5),
            margin=dict(l=10, r=10, t=50, b=10),
            hoverlabel=dict(bgcolor=WICKET_THEME['card_bg'], font_color=WICKET_THEME['text_light'])
        )
        fig.add_annotation(
            text="Real-time drone tracking",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(color=WICKET_THEME['text_light'], size=12),
            bgcolor=WICKET_THEME['card_bg']
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.info("Drone map rendered successfully")
    except Exception as e:
        st.error("Failed to render drone map. Please check your Mapbox token and internet connection.")
        logger.error(f"Drone map rendering failed: {str(e)}")
    st.dataframe(df[['timestamp', 'drone_id', 'latitude', 'longitude', 'altitude', 'status', 'severity']])

# Simulated Radar Surveillance
def simulate_radar_data(num_targets=30):
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
    try:
        fig = go.Figure()
        # Animated radar sweep
        theta = np.linspace(0, 360, 100)
        r = np.ones(100) * 0.7
        lon_sweep = 7 + r * np.cos(np.radians(theta))
        lat_sweep = 9 + r * np.sin(np.radians(theta))
        fig.add_trace(go.Scattermapbox(
            lon=lon_sweep,
            lat=lat_sweep,
            mode='lines',
            line=dict(color=WICKET_THEME['accent'], width=2),
            fill='toself',
            opacity=0.4,
            name='Radar Sweep',
            customdata=[0] * len(theta),
            hoverinfo='skip'
        ))
        # Radar targets
        fig.add_trace(go.Scattermapbox(
            lon=df['longitude'],
            lat=df['latitude'],
            mode='markers',
            marker=dict(
                size=8,  # Smaller size for 30 targets
                color=WICKET_THEME['text_light'],
                symbol='cross',
                opacity=0.8
            ),
            text=df['target_id'],
            hovertemplate="%{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>Altitude: %{customdata[0]:.0f}ft<br>Velocity: %{customdata[1]:.0f}kts<extra></extra>",
            customdata=df[['altitude', 'velocity']].values,
            name='Radar Targets'
        ))
        mapbox_token = st.secrets.get('MAPBOX_TOKEN', 'pk.eyJ1IjoiZ3JvazMiLCJhIjoiY2x6aG5sOHVrMDM3NjJrbzF0M3A0eTRsZyJ9._ZJqGa-3kT-Zv2Cto0L_2Q')
        fig.update_layout(
            mapbox=dict(
                style='streets-v12',
                center=dict(lat=9, lon=7),
                zoom=7,
                accesstoken=mapbox_token
            ),
            showlegend=True,
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            title=dict(text="Radar Surveillance", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5),
            margin=dict(l=10, r=10, t=50, b=10),
            hoverlabel=dict(bgcolor=WICKET_THEME['card_bg'], font_color=WICKET_THEME['text_light'])
        )
        fig.add_annotation(
            text="Radar sweep simulation",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(color=WICKET_THEME['text_light'], size=12),
            bgcolor=WICKET_THEME['card_bg']
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.info("Radar map rendered successfully")
    except Exception as e:
        st.error("Failed to render radar map. Please check your Mapbox token and internet connection.")
        logger.error(f"Radar map rendering failed: {str(e)}")
    st.dataframe(df[['timestamp', 'target_id', 'latitude', 'longitude', 'altitude', 'velocity']])

# Simulated ATC Monitoring
def simulate_atc_data(num_samples=30):
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
    try:
        fig = go.Figure()
        for anomaly in [False, True]:
            anomaly_df = df[df['anomaly'] == anomaly]
            if not anomaly_df.empty:
                fig.add_trace(go.Scattermapbox(
                    lon=anomaly_df['longitude'],
                    lat=anomaly_df['latitude'],
                    mode='markers',
                    marker=dict(
                        size=10,  # Smaller size for 30 targets
                        color=WICKET_THEME['error'] if anomaly else WICKET_THEME['success'],
                        symbol='airplane',
                        opacity=0.8
                    ),
                    text=anomaly_df['icao24'],
                    hovertemplate="%{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<br>Altitude: %{customdata[0]:.0f}ft<br>Velocity: %{customdata[1]:.0f}kts<extra></extra>",
                    customdata=anomaly_df[['altitude', 'velocity']].values,
                    name='Anomaly' if anomaly else 'Normal'
                ))
        mapbox_token = st.secrets.get('MAPBOX_TOKEN', 'pk.eyJ1IjoiZ3JvazMiLCJhIjoiY2x6aG5sOHVrMDM3NjJrbzF0M3A0eTRsZyJ9._ZJqGa-3kT-Zv2Cto0L_2Q')
        fig.update_layout(
            mapbox=dict(
                style='streets-v12',
                center=dict(lat=9, lon=7),
                zoom=7,
                accesstoken=mapbox_token
            ),
            showlegend=True,
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            title=dict(text="ATC Monitoring", font=dict(color=WICKET_THEME['text_light'], size=20), x=0.5),
            margin=dict(l=10, r=10, t=50, b=10),
            hoverlabel=dict(bgcolor=WICKET_THEME['card_bg'], font_color=WICKET_THEME['text_light'])
        )
        fig.add_annotation(
            text="ATC anomaly detection",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(color=WICKET_THEME['text_light'], size=12),
            bgcolor=WICKET_THEME['card_bg']
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.info("ATC map rendered successfully")
    except Exception as e:
        st.error("Failed to render ATC map. Please check your Mapbox token and internet connection.")
        logger.error(f"ATC map rendering failed: {str(e)}")
    st.dataframe(df[['timestamp', 'icao24', 'latitude', 'longitude', 'altitude', 'velocity', 'anomaly']])
    if st.session_state.flight_conflicts:
        st.subheader("Collision Risks")
        st.dataframe(pd.DataFrame(st.session_state.flight_conflicts))

# Simulated Threat Intelligence
def simulate_threat_intelligence(num_threats=30):
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

    st.sidebar.image("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/nama_logo.jpg")
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
