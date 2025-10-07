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
import sqlite3
import re
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logging.warning("bcrypt unavailable. Using insecure password storage.")
logging.basicConfig(level=logging.INFO, filename='guardianeye.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
def init_db():
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (username TEXT PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
            c.execute('SELECT username FROM users WHERE username = ?', ('guardian',))
            if not c.fetchone():
                admin_password = 'admin'
                hashed_pw = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') if BCRYPT_AVAILABLE else admin_password
                c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                          ('guardian', 'admin@guardianeye.com', hashed_pw))
                logger.info("Admin user created")
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        st.error("Database error. Try again later.")
def is_valid_email(email):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email)
def is_valid_password(password):
    return len(password) >= 8 and any(c.isupper() for c in password) and any(c.isdigit() for c in password)
WICKET_THEME = {
    "primary_bg": "#1E2A44",
    "accent": "#00D4FF",
    "text": "#E6E6FA",
    "text_light": "#FFFFFF",
    "card_bg": "rgba(30, 42, 68, 0.7)",
    "border": "#3B82F6",
    "button_bg": "#00D4FF",
    "button_text": "#0A0F2D",
    "error": "#FF4D4D",
    "success": "#00FF99"
}
def apply_wicket_css():
    css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap');
            .stApp {{
                background: url('https://images.stockcake.com/public/a/d/0/ad04b73f-08d2-4c89-bdcd-3cc8db5ed03f_large/cybernetic-eye-glows-stockcake.jpg');
                background-size: cover;
                background-position: center;
                background-color: {WICKET_THEME['primary_bg']};
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
                position: relative;
            }}
            .stApp::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(30, 42, 68, 0.3);
                z-index: -1;
            }}
            .card {{
                background: {WICKET_THEME['card_bg']};
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
            }}
            .stButton>button {{
                background: {WICKET_THEME['button_bg']};
                color: {WICKET_THEME['button_text']};
                border-radius: 15px;
                padding: 8px 20px;
                border: none;
                font-family: 'Roboto Mono', monospace;
                font-weight: 500;
            }}
            .stButton>button:hover {{
                background: {WICKET_THEME['accent']};
                box-shadow: 0 0 10px {WICKET_THEME['accent']};
            }}
            .plotly-graph-div {{
                background: {WICKET_THEME['card_bg']};
                border-radius: 8px;
                padding: 5px;
            }}
            h1, h2, h3 {{
                font-family: 'Roboto Mono', monospace;
                color: {WICKET_THEME['text_light']};
            }}
            .stSidebar .sidebar-content img {{
                width: 100px;
                margin: 0 auto;
                display: block;
            }}
            .auth-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                padding: 10px;
            }}
            .form-container {{
                background: {WICKET_THEME['card_bg']};
                border-radius: 8px;
                padding: 20px;
                width: 100%;
                max-width: 300px;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
            }}
            .stTextInput input, .stTextInput input:focus {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 5px;
                color: {WICKET_THEME['text']};
            }}
            .stTextInput input:focus {{
                border-color: {WICKET_THEME['accent']};
            }}
            .logo {{
                display: block;
                margin: 0 auto 10px;
                width: 100px;
            }}
            .forgot-password {{
                color: {WICKET_THEME['accent']};
                font-size: 0.8rem;
                text-decoration: none;
                margin: 5px 0;
                display: block;
            }}
            .debug-text {{
                color: {WICKET_THEME['success']};
                font-size: 1.2em;
                text-align: center;
                z-index: 10;
            }}
            .particles {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
            }}
            .particle {{
                position: absolute;
                background: {WICKET_THEME['accent']};
                border-radius: 50%;
                opacity: 0.5;
                animation: float 10s linear infinite;
            }}
            @keyframes float {{
                0% {{ transform: translateY(0); opacity: 0.5; }}
                100% {{ transform: translateY(-100vh); opacity: 0; }}
            }}
            .scanline {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 2px;
                background: {WICKET_THEME['accent']};
                opacity: 0.5;
                animation: scan 5s linear infinite;
                z-index: 1;
                pointer-events: none;
            }}
            @keyframes scan {{
                0% {{ top: 0; opacity: 0.5; }}
                100% {{ top: 100%; opacity: 0; }}
            }}
        </style>
        <div class="debug-text">GuardianEye: Rendering Dashboard</div>
        <div class="scanline"></div>
        <script>
            function createParticles() {{
                const particleContainer = document.createElement('div');
                particleContainer.className = 'particles';
                document.body.appendChild(particleContainer);
                for (let i = 0; i < 10; i++) {{
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    particle.style.width = Math.random() * 2 + 2 + 'px';
                    particle.style.height = particle.style.width;
                    particle.style.left = Math.random() * 100 + 'vw';
                    particle.style.animationDuration = (Math.random() * 5 + 5) + 's';
                    particleContainer.appendChild(particle);
                }}
            }}
            window.onload = createParticles;
        </script>
    """
    st.markdown(css, unsafe_allow_html=True)
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
if 'airports_data' not in st.session_state:
    st.session_state.airports_data = []
if 'panel_state' not in st.session_state:
    st.session_state.panel_state = 'sign_in'
init_db()
def render_auth_ui():
    if not BCRYPT_AVAILABLE:
        st.warning("Secure password hashing unavailable.")
    st.markdown(
        '<div class="auth-container">'
        f'<img src="https://images.stockcake.com/public/a/d/0/ad04b73f-08d2-4c89-bdcd-3cc8db5ed03f_large/cybernetic-eye-glows-stockcake.jpg" class="logo">'
        f'<div class="form-container">',
        unsafe_allow_html=True
    )
    if st.session_state.panel_state == 'sign_in':
        with st.form(key='sign_in_form'):
            st.markdown('<h2 style="text-align: center;">Sign In</h2>', unsafe_allow_html=True)
            username = st.text_input('Username', placeholder='Username')
            password = st.text_input('Password', type='password', placeholder='Password')
            st.markdown('<a href="#" class="forgot-password">Forgot password?</a>', unsafe_allow_html=True)
            submit = st.form_submit_button('Sign In')
            if submit:
                try:
                    with sqlite3.connect('users.db') as conn:
                        c = conn.cursor()
                        c.execute('SELECT password FROM users WHERE username = ?', (username,))
                        result = c.fetchone()
                        if result:
                            stored_password = result[0]
                            if BCRYPT_AVAILABLE and bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')) or not BCRYPT_AVAILABLE and password == stored_password:
                                st.session_state.authenticated = True
                                logger.info(f"Authenticated: {username}")
                                st.rerun()
                            else:
                                st.error('Invalid credentials')
                                logger.warning(f"Failed login: {username}")
                        else:
                            st.error('Invalid credentials')
                            logger.warning(f"Failed login: {username}")
                except sqlite3.Error as e:
                    st.error('Database error')
                    logger.error(f"Login error: {str(e)}")
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
                    st.error('All fields required')
                elif username.lower() == 'guardian':
                    st.error('Username "guardian" reserved')
                elif not is_valid_email(email):
                    st.error('Invalid email')
                elif not is_valid_password(password):
                    st.error('Password needs 8+ chars, uppercase, number')
                else:
                    try:
                        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') if BCRYPT_AVAILABLE else password
                        with sqlite3.connect('users.db') as conn:
                            c = conn.cursor()
                            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                                      (username, email, hashed_pw))
                            conn.commit()
                            st.success('Account created. Sign in.')
                            st.session_state.panel_state = 'sign_in'
                            logger.info(f"User registered: {username}")
                            st.rerun()
                    except sqlite3.IntegrityError:
                        st.error('Username or email exists')
                    except sqlite3.Error as e:
                        st.error('Database error')
                        logger.error(f"Sign-up error: {str(e)}")
        st.button('Switch to Sign In', on_click=lambda: st.session_state.update(panel_state='sign_in'))
    st.markdown('</div></div>', unsafe_allow_html=True)
def simulate_drone_data(num_drones=10):
    region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
    drones = []
    for i in range(num_drones):
        altitude = np.random.uniform(50, 1000)
        is_unauthorized = altitude < 400 and np.random.random() > 0.5
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
    logger.info(f"Simulated {len(drones)} drones")
    return drones
def display_drone_data():
    if not st.session_state.drone_results:
        st.warning("No drone data. Click 'Simulate Drones'.")
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
                marker=dict(size=8, color=WICKET_THEME['error'] if status == 'unidentified' else WICKET_THEME['success']),
                text=status_df['drone_id'],
                hoverinfo='text',
                name=status.capitalize()
            ))
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=9, lon=8),
                zoom=5
            ),
            showlegend=True,
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            title=dict(text="Drone Surveillance", font=dict(color=WICKET_THEME['text_light']))
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Failed to render drone map. Check internet connection.")
        logger.error(f"Drone map error: {str(e)}")
    st.dataframe(df[['timestamp', 'drone_id', 'latitude', 'longitude', 'altitude', 'status']])
def simulate_radar_data(num_targets=10):
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
        st.warning("No radar data. Click 'Simulate Radar'.")
        return
    df = pd.DataFrame(st.session_state.radar_data)
    try:
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lon=df['longitude'],
            lat=df['latitude'],
            mode='markers',
            marker=dict(size=6, color=WICKET_THEME['text_light']),
            text=df['target_id'],
            hoverinfo='text',
            name='Radar Targets'
        ))
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=9, lon=8),
                zoom=5
            ),
            showlegend=True,
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            title=dict(text="Radar Surveillance", font=dict(color=WICKET_THEME['text_light']))
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Failed to render radar map. Check internet connection.")
        logger.error(f"Radar map error: {str(e)}")
    st.dataframe(df[['timestamp', 'target_id', 'latitude', 'longitude', 'altitude', 'velocity']])
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
    logger.info(f"Simulated {len(data)} ATC records")
    return df.to_dict('records')
def display_atc_data():
    if not st.session_state.atc_results:
        st.warning("No ATC data. Click 'Simulate ATC'.")
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
                    marker=dict(size=8, color=WICKET_THEME['error'] if anomaly else WICKET_THEME['success']),
                    text=anomaly_df['icao24'],
                    hoverinfo='text',
                    name='Anomaly' if anomaly else 'Normal'
                ))
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=9, lon=8),
                zoom=5
            ),
            showlegend=True,
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            title=dict(text="ATC Monitoring", font=dict(color=WICKET_THEME['text_light']))
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Failed to render ATC map. Check internet connection.")
        logger.error(f"ATC map error: {str(e)}")
    st.dataframe(df[['timestamp', 'icao24', 'latitude', 'longitude', 'altitude', 'velocity', 'anomaly']])
    if st.session_state.flight_conflicts:
        st.subheader("Collision Risks")
        st.dataframe(pd.DataFrame(st.session_state.flight_conflicts))
def simulate_threat_intelligence(num_threats=10):
    threats = []
    for i in range(num_threats):
        threats.append({
            'timestamp': datetime.now(),
            'threat_id': f"THR{i:03d}",
            'description': f"Threat {i+1} near Nigerian airport",
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
        st.warning("No threat data. Click 'Simulate Threats'.")
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
        title="Threat Severity",
        xaxis_title="Severity",
        yaxis_title="Count",
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        font={'color': WICKET_THEME['text_light']}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['timestamp', 'threat_id', 'description', 'severity']])
def simulate_compliance_metrics():
    metrics = {
        'detection_rate': np.random.uniform(70, 95),
        'open_ports': np.random.randint(0, 10),
        'alerts': len(st.session_state.alert_log)
    }
    if metrics['open_ports'] > 5 or metrics['alerts'] > 3:
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
        st.warning("No compliance data. Click 'Simulate Compliance'.")
        return
    metrics = st.session_state.compliance_metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detection Rate", f"{metrics['detection_rate']:.1f}%")
    with col2:
        st.metric("Open Ports", metrics['open_ports'])
    with col3:
        st.metric("Alerts", metrics['alerts'])
def simulate_nmap_scan(target="192.168.1.1"):
    common_ports = {21: 'ftp', 22: 'ssh', 80: 'http', 443: 'https', 3306: 'mysql'}
    np.random.seed(42)
    scan_results = []
    for port, service in common_ports.items():
        state = 'open' if np.random.random() > 0.5 else 'closed'
        scan_results.append({'port': port, 'protocol': 'tcp', 'state': state, 'service': service})
    open_ports = len([r for r in scan_results if r['state'] == 'open'])
    st.session_state.compliance_metrics['open_ports'] = open_ports
    st.session_state.alert_log.append({
        'timestamp': datetime.now(),
        'type': 'Network Scan',
        'severity': 'medium',
        'details': f"Scanned {target}, found {open_ports} open ports"
    })
    logger.info(f"Simulated scan: {open_ports} open ports")
    return scan_results
def display_network_scan():
    if not st.session_state.scan_results:
        st.warning("No scan data. Click 'Simulate Scan'.")
        return
    df = pd.DataFrame(st.session_state.scan_results)
    st.dataframe(df[['port', 'protocol', 'state', 'service']])
def simulate_nigerian_airports():
    airports = [
        {"name": "Murtala Muhammed Int'l", "state": "Lagos", "lat": 6.5772, "lon": 3.3212, "icao": "DNMM"},
        {"name": "Nnamdi Azikiwe Int'l", "state": "Abuja", "lat": 9.0068, "lon": 7.2632, "icao": "DNAA"},
        {"name": "Mallam Aminu Kano Int'l", "state": "Kano", "lat": 12.0470, "lon": 8.5247, "icao": "DNKN"},
        {"name": "Port Harcourt Int'l", "state": "Rivers", "lat": 4.8772, "lon": 7.0161, "icao": "DNPO"},
        {"name": "Akanu Ibiam Int'l", "state": "Enugu", "lat": 6.4744, "lon": 7.5617, "icao": "DNEN"}
    ]
    for airport in airports:
        airport['threats'] = np.random.randint(0, 6)
        airport['vuln_level'] = 'low' if airport['threats'] < 2 else 'medium' if airport['threats'] < 4 else 'high'
        if airport['vuln_level'] == 'high':
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Airport Vulnerability',
                'severity': 'high',
                'details': f"High threats at {airport['name']} ({airport['state']})"
            })
    logger.info(f"Simulated {len(airports)} airports")
    return airports
def display_nigerian_airports():
    if not st.session_state.airports_data:
        st.warning("No airport data. Click 'Simulate Airports'.")
        return
    df = pd.DataFrame(st.session_state.airports_data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Airport Threats")
        st.dataframe(df[['name', 'state', 'icao', 'threats', 'vuln_level']])
    with col2:
        vuln_counts = df['vuln_level'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(labels=vuln_counts.index, values=vuln_counts.values, hole=0.3)])
        fig_pie.update_layout(title="Vulnerability Distribution", paper_bgcolor=WICKET_THEME['card_bg'], font={'color': WICKET_THEME['text_light']})
        st.plotly_chart(fig_pie, use_container_width=True)
    try:
        fig = go.Figure(go.Scattermapbox(
            lon=df['lon'],
            lat=df['lat'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=['#00FF99' if v == 'low' else '#FFA500' if v == 'medium' else '#FF4D4D' for v in df['vuln_level']],
                opacity=0.8
            ),
            text=df['icao'],
            textposition="top center",
            hoverinfo='text',
            hovertext=df['name'] + '<br>Threats: ' + df['threats'].astype(str) + '<br>Vuln: ' + df['vuln_level']
        ))
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=9, lon=8),
                zoom=5
            ),
            title="Nigerian Airports Vulnerability Map",
            paper_bgcolor=WICKET_THEME['card_bg'],
            plot_bgcolor=WICKET_THEME['card_bg'],
            font={'color': WICKET_THEME['text_light']}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="debug-text">Map Rendered Successfully</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error("Failed to render airport map. Check internet connection.")
        logger.error(f"Airport map error: {str(e)}")
def generate_report():
    buffer = io.BytesIO()
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("GuardianEye Nigeria Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated: {datetime.now()}", styles['Normal']))
    if st.session_state.alert_log:
        elements.append(Paragraph("Recent Alerts", styles['Heading2']))
        alert_data = [[str(a['timestamp']), a['type'], a['severity'], a['details']] for a in st.session_state.alert_log]
        alert_table = Table([['Timestamp', 'Type', 'Severity', 'Details']] + alert_data)
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(alert_table)
    if st.session_state.airports_data:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Airport Vulnerabilities", styles['Heading2']))
        airport_data = [[a['name'], a['state'], a['icao'], a['threats'], a['vuln_level']] for a in st.session_state.airports_data]
        airport_table = Table([['Airport', 'State', 'ICAO', 'Threats', 'Vuln Level']] + airport_data)
        airport_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(airport_table)
    doc.build(elements)
    buffer.seek(0)
    return buffer
def main():
    apply_wicket_css()
    if not st.session_state.authenticated:
        render_auth_ui()
        return
    st.sidebar.image("https://images.stockcake.com/public/a/d/0/ad04b73f-08d2-4c89-bdcd-3cc8db5ed03f_large/cybernetic-eye-glows-stockcake.jpg", caption="GuardianEye")
    st.markdown('<div style="text-align:center;"><h1>GUARDIANEYE</h1><p style="color:#00D4FF;">Nigerian Airspace Security</p></div>', unsafe_allow_html=True)
    page = st.sidebar.selectbox("Select Feature", [
        "🏠 Dashboard",
        "🔍 Network Scan",
        "🚁 Drone Detection",
        "📡 Radar Surveillance",
        "✈️ ATC Monitoring",
        "⚠️ Threat Intelligence",
        "🛡️ Compliance Monitoring",
        "🗺️ Airport Security"
    ])
    if page == "🏠 Dashboard":
        st.markdown('<div class="card"><h2>Security Dashboard</h2><p>Monitor Nigerian airspace.</p></div>', unsafe_allow_html=True)
        if st.button("Run All Simulations"):
            st.session_state.scan_results = simulate_nmap_scan()
            st.session_state.drone_results = simulate_drone_data()
            st.session_state.radar_data = simulate_radar_data()
            st.session_state.atc_results = simulate_atc_data()
            st.session_state.threats = simulate_threat_intelligence()
            st.session_state.compliance_metrics = simulate_compliance_metrics()
            st.session_state.airports_data = simulate_nigerian_airports()
            st.success("Simulations completed.")
        if st.session_state.alert_log:
            st.subheader("Recent Alerts")
            st.dataframe(pd.DataFrame(st.session_state.alert_log[-5:]))
        if st.button("Download Report"):
            buffer = generate_report()
            b64 = base64.b64encode(buffer.getvalue()).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="guardianeye_report.pdf">Download PDF</a>', unsafe_allow_html=True)
    elif page == "🔍 Network Scan":
        st.markdown('<div class="card"><h2>Network Scan</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Scan"):
            st.session_state.scan_results = simulate_nmap_scan()
        display_network_scan()
    elif page == "🚁 Drone Detection":
        st.markdown('<div class="card"><h2>Drone Detection</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Drones"):
            st.session_state.drone_results = simulate_drone_data()
        display_drone_data()
    elif page == "📡 Radar Surveillance":
        st.markdown('<div class="card"><h2>Radar Surveillance</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Radar"):
            st.session_state.radar_data = simulate_radar_data()
        display_radar_data()
    elif page == "✈️ ATC Monitoring":
        st.markdown('<div class="card"><h2>ATC Monitoring</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate ATC"):
            st.session_state.atc_results = simulate_atc_data()
        display_atc_data()
    elif page == "⚠️ Threat Intelligence":
        st.markdown('<div class="card"><h2>Threat Intelligence</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Threats"):
            st.session_state.threats = simulate_threat_intelligence()
        display_threat_intelligence()
    elif page == "🛡️ Compliance Monitoring":
        st.markdown('<div class="card"><h2>Compliance Monitoring</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Compliance"):
            st.session_state.compliance_metrics = simulate_compliance_metrics()
        display_compliance_metrics()
    elif page == "🗺️ Airport Security":
        st.markdown('<div class="card"><h2>Nigerian Airport Security</h2></div>', unsafe_allow_html=True)
        if st.button("Simulate Airports"):
            st.session_state.airports_data = simulate_nigerian_airports()
        display_nigerian_airports()
if __name__ == "__main__":
    main()
