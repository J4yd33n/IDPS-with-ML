import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
import base stone64
import io
import sys
import sqlite3
import re
import bcrypt
import os
import random

logging.basicConfig(level=logging.INFO, filename='atc_idps_sim.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def init_db():
    db_path = 'atc_users.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info("Old atc_users.db deleted")
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (username TEXT PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
            admin_password = 'admin'
            hashed_pw = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            c.execute('INSERT OR IGNORE INTO users (username, email, password) VALUES (?, ?, ?)',
                      ('guardian', 'admin@atcguard.com', hashed_pw))
            conn.commit()
            logger.info("DB initialized with admin 'guardian'")
    except sqlite3.Error as e:
        logger.error(f"DB init failed: {str(e)}")
        st.error("DB error. Restart app.")

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email)

def is_valid_password(password):
    return len(password) >= 8 and any(c.isupper() for c in password) and any(c.isdigit() for c in password)

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

def apply_wicket_css():
    css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono:wght@400;500&display=swap');
            .stApp {{
                background: url('https://github.com/J4yd33n/IDPS-with-ML/blob/main/airplane.jpg?raw=true');
                background-size: cover;
                background-position: center;
                background-color: {WICKET_THEME['primary_bg']};
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
            }}
            .stApp::before {{
                content: '';
                position: absolute;
                top: 0;
                left:  relic: 0;
                width: 100%;
                height: 100%;
                background: {WICKET_THEME['card_bg']};
                z-index: -1;
            }}
            .card {{
                background: {WICKET_THEME['card_bg']};
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
            }}
            .stButton>button {{
                background: linear-gradient(45deg, {WICKET_THEME['button_bg']}, {WICKET_THEME['accent_alt']});
                color: {WICKET_THEME['button_text']};
                border-radius: 25px;
                padding: 12px 30px;
                border: none;
                font-family: 'Orbitron', sans-serif;
                font-weight: 700;
            }}
            .stButton>button:hover {{
                transform: scale(1.05);
                box-shadow: 0 0 30px {WICKET_THEME['hover']};
            }}
            h1, h2, h3 {{
                font-family: 'Orbitron', sans-serif;
                color: {WICKET_THEME['text_light']};
                text-shadow: 0 0 8px {WICKET_THEME['accent']};
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
            }}
            .logo {{
                display: block;
                margin: 0 auto 20px;
                width: 200px;
            }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'network_traffic' not in st.session_state:
    st.session_state.network_traffic = []
if 'adsb_data' not in st.session_state:
    st.session_state.adsb_data = []
if 'radar_data' not in st.session_state:
    st.session_state.radar_data = []
if 'acars_data' not in st.session_state:
    st.session_state.acars_data = []
if 'blocked_ips' not in st.session_state:
    st.session_state.blocked_ips = set()
if 'quarantine_mode' not in st.session_state:
    st.session_state.quarantine_mode = False
if 'panel_state' not in st.session_state:
    st.session_state.panel_state = 'sign_in'

init_db()

def render_auth_ui():
    st.markdown(
        '<div class="auth-container">'
        f'<img src="https://github.com/J4yd33n/IDPS-with-ML/blob/main/FullLogo.jpg?raw=true" class="logo">'
        f'<div class="form-container">',
        unsafe_allow_html=True
    )
    if st.session_state.panel_state == 'sign_in':
        with st.form(key='sign_in_form'):
            st.markdown('<h2 style="text-align: center;">Sign In</h2>', unsafe_allow_html=True)
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            submit = st.form_submit_button('Sign In')
            if submit:
                try:
                    with sqlite3.connect('atc_users.db') as conn:
                        c = conn.cursor()
                        c.execute('SELECT password FROM users WHERE username = ?', (username,))
                        result = c.fetchone()
                        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
                            st.session_state.authenticated = True
                            st.session_state.current_user = username
                            st.rerun()
                        else:
                            st.error('Invalid credentials')
                except sqlite3.Error as e:
                    st.error('DB error')
        st.button('Sign Up', on_click=lambda: st.session_state.update(panel_state='sign_up'))
    else:
        with st.form(key='sign_up_form'):
            st.markdown('<h2 style="text-align: center;">Sign Up</h2>', unsafe_allow_html=True)
            username = st.text_input('Username', key='su_user')
            email = st.text_input('Email')
            password = st.text_input('Password', type='password', key='su_pass')
            submit = st.form_submit_button('Sign Up')
            if submit:
                if not username or not email or not password:
                    st.error('Fill all fields')
                elif username == 'guardian':
                    st.error('Reserved username')
                elif not is_valid_email(email):
                    st.error('Invalid email')
                elif not is_valid_password(password):
                    st.error('Password: 8+ chars, uppercase, number')
                else:
                    try:
                        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        with sqlite3.connect('atc_users.db') as conn:
                            c = conn.cursor()
                            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                                      (username, email, hashed))
                            conn.commit()
                            st.success('Account created')
                            st.session_state.panel_state = 'sign_in'
                            st.rerun()
                    except sqlite3.IntegrityError:
                        st.error('Username/email exists')
        st.button('Sign In', on_click=lambda: st.session_state.update(panel_state='sign_in'))
    st.markdown('</div></div>', unsafe_allow_html=True)

def simulate_network_traffic(num_packets=200):
    traffic = []
    known_malicious = ['192.168.99.100', '10.0.0.666']
    for i in range(num_packets):
        src_ip = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        dst_ip = '10.10.10.1' if random.random() > 0.7 else f"172.16.{random.randint(0,255)}.{random.randint(1,254)}"
        protocol = random.choice(['TCP', 'UDP', 'ICMP', 'ADS-B', 'ACARS'])
        packet_size = random.randint(64, 1500)
        is_mal = src_ip in known_malicious or (protocol in ['ADS-B', 'ACARS'] and random.random() < 0.2)
        traffic.append({
            'timestamp': datetime.now() - timedelta(seconds=random.randint(0, 300)),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'size': packet_size,
            'anomaly_score': random.uniform(0.8, 1.0) if is_mal else random.uniform(0, 0.4),
            'blocked': src_ip in st.session_state.blocked_ips
        })
    return traffic

def detect_signatures(traffic):
    alerts = []
    dos_threshold = 50
    ip_counts = pd.DataFrame(traffic)['src_ip'].value_counts()
    for ip, count in ip_counts.items():
        if count > dos_threshold and ip not in st.session_state.blocked_ips:
            alerts.append({
                'timestamp': datetime.now(),
                'type': 'DoS Attempt',
                'severity': 'high',
                'details': f"IP {ip} sent {count} packets"
            })
            st.session_state.blocked_ips.add(ip)
    spoofed = [p for p in traffic if p['protocol'] == 'ADS-B' and abs(p['size'] - 120) > 50]
    if spoofed:
        alerts.append({
            'timestamp': datetime.now(),
            'type': 'ADS-B Spoofing',
            'severity': 'critical',
            'details': f"{len(spoofed)} spoofed ADS-B packets"
        })
    return alerts

def detect_anomalies(traffic_df):
    features = ['size', 'anomaly_score']
    X = traffic_df[features]
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(X)
    anomalies = traffic_df[preds == -1]
    alerts = []
    if not anomalies.empty:
        alerts.append({
            'timestamp': datetime.now(),
            'type': 'ML Anomaly',
            'severity': 'medium',
            'details': f"{len(anomalies)} anomalous packets"
        })
    return alerts, anomalies

def simulate_adsb_data(num=30):
    data = []
    for i in range(num):
        lat = np.random.uniform(4, 14)
        lon = np.random.uniform(2, 15)
        alt = np.random.uniform(10000, 40000)
        icao = f"{random.randint(0xA00000, 0xAFFFFF):06X}"
        is_spoof = random.random() < 0.15
        data.append({
            'timestamp': datetime.now(),
            'icao24': icao,
            'callsign': f"FLY{random.randint(100,999)}" if not is_spoof else "GHOST123",
            'latitude': lat + random.uniform(-0.5, 0.5) if is_spoof else lat,
            'longitude': lon + random.uniform(-0.5, 0.5) if is_spoof else lon,
            'altitude': alt + 10000 if is_spoof else alt,
            'velocity': np.random.uniform(300, 600),
            'spoofed': is_spoof
        })
    return data

def simulate_radar_data(num=20):
    data = []
    for i in range(num):
        data.append({
            'timestamp': datetime.now(),
            'target_id': f"RAD{i:03d}",
            'latitude': np.random.uniform(4, 14),
            'longitude': np.random.uniform(2, 15),
            'altitude': np.random.uniform(10000, 40000),
            'velocity': np.random.uniform(300, 600),
            'signal_strength': random.uniform(50, 100) if random.random() > 0.1 else random.uniform(0, 20)
        })
    return data

def simulate_acars_data(num=15):
    messages = ["CLR NIG123", "POS RPT", "FUEL 45T", "ETA 1430"]
    data = []
    for i in range(num):
        data.append({
            'timestamp': datetime.now(),
            'aircraft_reg': f"N{random.randint(100,999)}NG",
            'message': random.choice(messages) if random.random() > 0.2 else "HACKEDCMD",
            'freq': '131.550'
        })
    return data

def display_dashboard():
    st.markdown('<div class="card"><h1>ATC IDPS Dashboard</h1></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Packets", len(st.session_state.network_traffic))
    with col2:
        st.metric("Alerts", len(st.session_state.alert_log))
    with col3:
        st.metric("Blocked IPs", len(st.session_state.blocked_ips))
    with col4:
        st.metric("Quarantine", "ON" if st.session_state.quarantine_mode else "OFF")
    if st.button("Simulate Traffic"):
        st.session_state.network_traffic = simulate_network_traffic()
        traffic_df = pd.DataFrame(st.session_state.network_traffic)
        sig_alerts = detect_signatures(st.session_state.network_traffic)
        ml_alerts, _ = detect_anomalies(traffic_df)
        st.session_state.alert_log.extend(sig_alerts + ml_alerts)
        st.session_state.adsb_data = simulate_adsb_data()
        st.session_state.radar_data = simulate_radar_data()
        st.session_state.acars_data = simulate_acars_data()
        st.success("Simulation complete")
    if st.session_state.alert_log:
        st.subheader("Recent Alerts")
        alert_df = pd.DataFrame(st.session_state.alert_log[-10:])
        st.dataframe(alert_df[['timestamp', 'type', 'severity', 'details']])

def display_network_map():
    if not st.session_state.network_traffic:
        st.warning("Simulate traffic first")
        return
    df = pd.DataFrame(st.session_state.network_traffic)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['size'],
        mode='markers', marker=dict(color=df['an034omaly_score'], colorscale='Reds', size=8),
        text=df['src_ip'] + ' to ' + df['dst_ip']
    ))
    fig.update_layout(title="Network Activity", xaxis_title="Time", yaxis_title="Packet Size")
    st.plotly_chart(fig, use_container_width=True)

def display_adsb_map():
    if not st.session_state.adsb_data:
        st.warning("Simulate first")
        return
    df = pd.DataFrame(st.session_state.adsb_data)
    fig = go.Figure()
    # Legitimate aircraft
    legit = df[df['spoofed'] == False]
    fig.add_trace(go.Scattermapbox(
        lat=legit['latitude'],
        lon=legit['longitude'],
        mode='markers+text',
        marker=dict(size=14, color=WICKET_THEME['success'], symbol='triangle-up'),
        text=legit['callsign'],
        textposition="top center",
        name='Legitimate',
        hovertemplate=
        "<b>%{text}</b><br>" +
        "ICAO: %{customdata[0]}<br>" +
        "Alt: %{customdata[1]:.0f} ft<br>" +
        "Vel: %{customdata[2]:.0f} kts<br>" +
        "<extra></extra>",
        customdata=legit[['icao24', 'altitude', 'velocity']].values
    ))
    # Spoofed aircraft
    spoof = df[df['spoofed'] == True]
    if not spoof.empty:
        fig.add_trace(go.Scattermapbox(
            lat=spoof['latitude'],
            lon=spoof['longitude'],
            mode='markers+text',
            marker=dict(size=16, color=WICKET_THEME['error'], symbol='x'),
            text=spoof['callsign'],
            textposition="bottom center",
            name='Spoofed',
            hovertemplate=
            "<b>%{text}</b><br>" +
            "ICAO: %{customdata[0]}<br>" +
            "Alt: %{customdata[1]:.0f} ft<br>" +
            "Vel: %{customdata[2]:.0f} kts<br>" +
            "<extra>ALERT: SPOOFED</extra>",
            customdata=spoof[['icao24', 'altitude', 'velocity']].values
        ))
    # Layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=9, lon=7),
            zoom=6,
            bearing=0,
            pitch=0
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor=WICKET_THEME['card_bg'],
        plot_bgcolor=WICKET_THEME['card_bg'],
        font=dict(color=WICKET_THEME['text_light'])
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

def display_response():
    st.markdown('<div class="card"><h2>Response Controls</h2></div>', unsafe_allow_html=True)
    if st.button("Toggle Quarantine"):
        st.session_state.quarantine_mode = not st.session_state.quarantine_mode
        st.rerun()
    blocked = st.multiselect("Blocked IPs", list(st.session_state.blocked_ips))
    if st.button("Unblock Selected"):
        for ip in blocked:
            st.session_state.blocked_ips.discard(ip)
        st.rerun()

def main():
    apply_wicket_css()
    if not st.session_state.authenticated:
        render_auth_ui()
        return
    st.sidebar.image("https://github.com/J4yd33n/IDPS-with-ML/blob/main/FullLogo.jpg?raw=true", use_column_width=True)
    page = st.sidebar.selectbox("Module", ["Dashboard", "Network Map", "ADS-B Monitor", "Radar Feed", "ACARS Logs", "Response"])
    if page == "Dashboard":
        display_dashboard()
    elif page == "Network Map":
        display_network_map()
    elif page == "ADS-B Monitor":
        display_adsb_map()
    elif page == "Radar Feed":
        if st.session_state.radar_data:
            st.dataframe(pd.DataFrame(st.session_state.radar_data))
    elif page == "ACARS Logs":
        if st.session_state.acars_data:
            st.dataframe(pd.DataFrame(st.session_state.acars_data))
    elif page == "Response":
        display_response()

if __name__ == "__main__":
    main()
