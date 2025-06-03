import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import os
import sqlite3
import base64
import io
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import re
import time
import threading
from scipy.interpolate import interp1d
from geopy.distance import geodesic

# Check for optional dependencies
try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, filename='nama_idps.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NSL-KDD columns
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
]
LOW_IMPORTANCE_FEATURES = [
    'num_outbound_cmds', 'is_host_login', 'su_attempted', 'urgent', 'land',
    'num_access_files', 'num_shells', 'root_shell', 'num_failed_logins',
    'num_file_creations', 'num_root'
]
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

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
            @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
            
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
            
            .css-1d391kg {{
                background: {WICKET_THEME['card_bg']};
                backdrop-filter: blur(10px);
                color: {WICKET_THEME['text']};
                padding: 20px;
                border-right: 2px solid {WICKET_THEME['accent']};
                animation: slideInLeft 0.5s ease-out;
            }}
            
            @keyframes slideInLeft {{
                from {{ transform: translateX(-100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            
            .sidebar .sidebar-content {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            
            .sidebar-item {{
                display: flex;
                align-items: center;
                padding: 12px 20px;
                background: {WICKET_THEME['card_bg']};
                border-radius: 8px;
                color: {WICKET_THEME['text']};
                text-decoration: none;
                transition: all 0.3s ease;
                font-family: 'Orbitron', sans-serif;
                font-size: 16px;
            }}
            .sidebar-item:hover {{
                background: {WICKET_THEME['hover']};
                transform: translateX(10px);
                box-shadow: 0 0 15px {WICKET_THEME['hover']};
            }}
            .sidebar-item i {{
                margin-right: 10px;
                color: {WICKET_THEME['accent']};
            }}
            
            .main .block-container {{
                padding: 30px;
                max-width: 1400px;
                margin: auto;
                animation: fadeInUp 0.7s ease-out;
            }}
            
            @keyframes fadeInUp {{
                0% {{ opacity: 0; transform: translateY(20px); }}
                100% {{ opacity: 1; transform: translateY(0); }}
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
            
            .stTextInput>div>input, .stSelectbox>div>select {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 8px;
                padding: 12px;
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }}
            .stTextInput input:focus {{
                border-color: {WICKET_THEME['accent']};
                box-shadow: 0 0 10px {WICKET_THEME['accent']};
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
                position: relative;
                overflow: hidden;
            }}
            .stButton>button:hover {{
                transform: scale(1.05);
                box-shadow: 0 0 20px {WICKET_THEME['hover']};
                background: linear-gradient(45deg, {WICKET_THEME['hover']}, {WICKET_THEME['accent_alt']});
            }}
            .stButton>button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.2),
                    transparent
                );
                transition: 0.5s;
            }}
            .stButton>button:hover::before {{
                left: 100%;
            }}
            
            .plotly-graph-div {{
                background: {WICKET_THEME['card_bg']};
                border-radius: 12px;
                padding: 10px;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
            }}
            
            .logo-image {{
                width: 100%;
                max-width: 180px;
                height: auto;
                margin-bottom: 20px;
                filter: drop-shadow(0 0 10px {WICKET_THEME['accent']});
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            
            h1, h2, h3 {{
                font-family: 'Orbitron', sans-serif;
                color: {WICKET_THEME['text_light']};
                text-shadow: 0 0 8px {WICKET_THEME['accent']};
            }}
            
            .stAlert {{
                background: {WICKET_THEME['error']};
                color: {WICKET_THEME['text_light']};
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 0 10px {WICKET_THEME['error']};
                animation: shake 0.5s;
            }}
            
            @keyframes shake {{
                0%, 100% {{ transform: translateX(0); }}
                25% {{ transform: translateX(-5px); }}
                75% {{ transform: translateX(5px); }}
            }}
            
            p, li, div, span {{
                color: {WICKET_THEME['text']};
                font-family: 'Roboto Mono', monospace;
            }}
            
            .auth-container {{
                background: url('https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/airplane.jpg') no-repeat center center fixed;
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
                animation: zoomIn 0.8s ease-in-out;
            }}
            
            @keyframes zoomIn {{
                0% {{ transform: scale(0.8); opacity: 0; }}
                100% {{ transform: scale(1); opacity: 1; }}
            }}
            
            .auth-form {{
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }}
            
            .auth-form h2 {{
                font-family: 'Orbitron', sans-serif;
                color: {WICKET_THEME['text_light']};
                margin-bottom: 1.5rem;
                text-shadow: 0 0 8px {WICKET_THEME['accent']};
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
                transition: all 0.3s ease;
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
                letter-spacing: 1px;
                cursor: pointer;
                margin-top: 1.5rem;
                transition: all 0.3s ease;
                box-shadow: 0 0 15px {WICKET_THEME['button_bg']};
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
                transition: color 0.3s ease;
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
            
            .custom-spinner {{
                border: 4px solid {WICKET_THEME['card_bg']};
                border-top: 4px solid {WICKET_THEME['accent']};
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: auto;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            @media (max-width: 768px) {{
                .main .block-container {{
                    padding: 15px;
                }}
                .card {{
                    padding: 15px;
                }}
                .sidebar-item {{
                    font-size: 14px;
                    padding: 10px;
                }}
            }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'compliance_metrics' not in st.session_state:
    st.session_state.compliance_metrics = {'detection_rate': 0, 'open_ports': 0, 'alerts': 0}
if 'user_activity' not in st.session_state:
    st.session_state.user_activity = {}
if 'equipment_status' not in st.session_state:
    st.session_state.equipment_status = []
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'atc_results' not in st.session_state:
    st.session_state.atc_results = []
if 'atc_anomalies' not in st.session_state:
    st.session_state.atc_anomalies = pd.DataFrame()
if 'threats' not in st.session_state:
    st.session_state.threats = []
if 'drone_results' not in st.session_state:
    st.session_state.drone_results = []
if 'radar_data' not in st.session_state:
    st.session_state.radar_data = []
if 'flight_conflicts' not in st.session_state:
    st.session_state.flight_conflicts = []
if 'optimized_routes' not in st.session_state:
    st.session_state.optimized_routes = []

# User database setup
def setup_user_db():
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_activity (
        username TEXT,
        timestamp TEXT,
        action TEXT
    )''')
    c.execute("SELECT username FROM users WHERE username = ?", ('nama',))
    if not c.fetchone() and BCRYPT_AVAILABLE:
        default_password = 'admin'
        hashed = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt())
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('nama', hashed))
            conn.commit()
            logger.info("Default user 'nama' created successfully")
        except sqlite3.IntegrityError:
            logger.error("Failed to create default user: username already exists")
    conn.commit()
    conn.close()

def register_user(username, password):
    if not BCRYPT_AVAILABLE:
        logger.error("Cannot register user: bcrypt module is missing")
        return False
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    conn.close()
    return True

def authenticate_user(username, password):
    if not BCRYPT_AVAILABLE:
        logger.error("Authentication disabled: bcrypt module is missing")
        st.error("Authentication disabled: bcrypt module is missing")
        return False
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_password = result[0]
        try:
            return bcrypt.checkpw(password.encode('utf-8'), stored_password)
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    return False

def log_user_activity(username, action):
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    c.execute("INSERT INTO user_activity (username, timestamp, action) VALUES (?, ?, ?)",
              (username, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action))
    conn.commit()
    conn.close()
    logger.info(f"User: {username}, Action: {action}")

# Preprocessing
def preprocess_data(df, label_encoders, le_class, is_train=True):
    try:
        df = df.copy()
        df.fillna({'protocol_type': 'missing', 'service': 'missing', 'flag': 'missing'}, inplace=True)
        df.fillna(0, inplace=True)
        
        numeric_cols = [col for col in NSL_KDD_COLUMNS if col not in CATEGORICAL_COLS + ['class'] + LOW_IMPORTANCE_FEATURES]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('', 'missing')
            if is_train:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
            else:
                unseen_mask = ~df[col].isin(label_encoders[col].classes_)
                df.loc[unseen_mask, col] = 'unknown'
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                df[col] = label_encoders[col].transform(df[col])
        
        if 'class' in df.columns:
            df['class'] = df['class'].astype(str)
            if is_train:
                le_class = LabelEncoder()
                df['class'] = le_class.fit_transform(df['class'])
            else:
                valid_classes = np.append(le_class.classes_, 'unknown')
                df['class'] = df['class'].apply(lambda x: x if x in valid_classes else 'unknown')
                if 'unknown' not in le_class.classes_:
                    le_class.classes_ = np.append(le_class.classes_, 'unknown')
                df['class'] = le_class.transform(df['class'])
        
        df = df.drop(columns=[col for col in LOW_IMPORTANCE_FEATURES if col in df.columns], errors='ignore')
        return df, label_encoders, le_class
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        st.error(f"Preprocessing error: {str(e)}")
        return None, label_encoders, le_class

# Radar Surveillance
def simulate_radar_data(num_targets=5, region=None):
    try:
        if region is None:
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
        radar_data = []
        for i in range(num_targets):
            radar_data.append({
                'target_id': f"RAD{i:03d}",
                'timestamp': pd.Timestamp.now(),
                'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
                'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
                'altitude': np.random.uniform(0, 40000),
                'velocity': np.random.uniform(0, 600),
                'heading': np.random.uniform(0, 360),
                'source': 'radar'
            })
        log_user_activity("system", f"Simulated radar data for {num_targets} targets")
        return radar_data
    except Exception as e:
        logger.error(f"Radar simulation error: {str(e)}")
        st.error(f"Radar simulation error: {str(e)}")
        return []

def merge_radar_adsb(radar_data, adsb_data):
    try:
        radar_df = pd.DataFrame(radar_data)
        adsb_df = pd.DataFrame(adsb_data)
        if not radar_df.empty:
            radar_df['icao24'] = radar_df['target_id']
        if not adsb_df.empty:
            adsb_df['target_id'] = adsb_df['icao24']
            adsb_df['source'] = 'ads-b'
            adsb_df['heading'] = np.random.uniform(0, 360, len(adsb_df))  # Simulated heading
        combined = pd.concat([radar_df, adsb_df], ignore_index=True)
        return combined.to_dict('records')
    except Exception as e:
        logger.error(f"Radar-ADS-B merge error: {str(e)}")
        st.error(f"Radar-ADS-B merge error: {str(e)}")
        return adsb_data

def display_radar(data):
    try:
        df = pd.DataFrame(data)
        if df.empty:
            st.warning("No radar data available")
            return None
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
            name='Radar Sweep',
            subplot='polar'
        ))
        # Aircraft positions
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            fig.add_trace(go.Scattergeo(
                lon=source_df['longitude'],
                lat=source_df['latitude'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=WICKET_THEME['accent'] if source == 'ads-b' else WICKET_THEME['error'],
                    symbol='circle',
                    line=dict(width=1, color=WICKET_THEME['text']),
                    opacity=0.8
                ),
                text=source_df['target_id'],
                hoverinfo='text',
                name=source.capitalize()
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
            margin=dict(l=10, r=10, t=50, b=10),
            title=dict(
                text="Real-Time Radar Surveillance",
                font=dict(color=WICKET_THEME['text_light'], size=20),
                x=0.5,
                xanchor='center'
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Radar display error: {str(e)}")
        st.error(f"Radar display error: {str(e)}")
        return None

def periodic_radar_update(num_targets, region, interval=60):
    while 'radar_running' in st.session_state and st.session_state.radar_running:
        radar_data = simulate_radar_data(num_targets, region)
        adsb_data = st.session_state.atc_results
        combined_data = merge_radar_adsb(radar_data, adsb_data)
        st.session_state.radar_data = combined_data
        time.sleep(interval)

# Real-time NMAP scan
def run_nmap_scan(target, scan_type, port_range, custom_args):
    try:
        if not NMAP_AVAILABLE:
            raise ImportError("python-nmap library is not installed.")
        nm = nmap.PortScannerAsync()
        scan_args = {'TCP SYN': '-sS', 'TCP Connect': '-sT', 'UDP': '-sU'}
        args = f"{scan_args[scan_type]} {custom_args}"
        
        scan_results = []
        
        def scan_callback(host, scan_data):
            if scan_data and 'scan' in scan_data:
                for host in scan_data['scan']:
                    for proto in scan_data['scan'][host].all_protocols():
                        for port in scan_data['scan'][host][proto].keys():
                            state = scan_data['scan'][host][proto][port]['state']
                            service = scan_data['scan'][host][proto][port].get('name', 'unknown')
                            scan_results.append({
                                'port': port,
                                'protocol': proto,
                                'state': state,
                                'service': service
                            })
                st.session_state.scan_results = scan_results
        
        nm.scan(target, port_range, arguments=args, callback=scan_callback)
        with st.spinner("Scanning in progress..."):
            while nm.still_scanning():
                time.sleep(0.1)
        
        log_user_activity("system", f"Real-time NMAP scan on {target}")
        return scan_results
    except Exception as e:
        logger.error(f"NMAP scan error: {str(e)}")
        st.warning(f"NMAP scan error: {str(e)}. Using simulated scan.")
        return simulate_nmap_scan(target, scan_type, port_range)

def periodic_nmap_scan(target, scan_type, port_range, custom_args, interval=300):
    while 'nmap_running' in st.session_state and st.session_state.nmap_running:
        results = run_nmap_scan(target, scan_type, port_range, custom_args)
        st.session_state.scan_results = results
        time.sleep(interval)

# Simulated NMAP scan
def simulate_nmap_scan(target, scan_type, port_range):
    try:
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
        log_user_activity("system", f"Simulated NMAP scan on {target}")
        return scan_results
    except Exception as e:
        logger.error(f"NMAP simulation error: {str(e)}")
        st.error(f"NMAP simulation error: {str(e)}")
        return []

# Enhanced ADS-B Data Fetching
@st.cache_data(ttl=60)
def fetch_adsb_data(num_samples=10, region=None):
    try:
        if region is None:
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
        username = st.secrets.get('OPENSKY_USERNAME', os.getenv('OPENSKY_USERNAME'))
        password = st.secrets.get('OPENSKY_PASSWORD', os.getenv('OPENSKY_PASSWORD'))
        params = {
            'lamin': region['lat_min'], 'lomin': region['lon_min'],
            'lamax': region['lat_max'], 'lomax': region['lon_max']
        }
        url = "https://opensky-network.org/api/states/all"
        for attempt in range(3):
            if username and password:
                response = requests.get(url, auth=(username, password), params=params, timeout=10)
            else:
                response = requests.get(url, params=params, timeout=10)
                logger.info("No credentials provided; using anonymous ADS-B access")
            if response.status_code == 200:
                data = response.json()['states']
                if not data:
                    logger.warning("No ADS-B data in region. Using simulated data.")
                    return simulate_aviation_traffic(num_samples, region)
                data = data[:num_samples]
                airports = ['DNMM', 'DNAA', 'DNKN', 'DNPO']
                adsb_records = []
                for state in data:
                    adsb_records.append({
                        'timestamp': pd.Timestamp.now(),
                        'icao24': state[0] if state[0] else 'unknown',
                        'protocol_type': 'ads-b',
                        'service': 'flight_data',
                        'src_bytes': np.random.randint(100, 1000),
                        'dst_bytes': np.random.randint(100, 1000),
                        'airport_code': np.random.choice(airports),
                        'duration': np.random.randint(0, 100),
                        'flag': 'SF',
                        'count': np.random.randint(1, 10),
                        'srv_count': np.random.randint(1, 10),
                        'serror_rate': 0.0,
                        'srv_serror_rate': 0.0,
                        'rerror_rate': 0.0,
                        'srv_rerror_rate': 0.0,
                        'same_srv_rate': 1.0,
                        'diff_srv_rate': 0.0,
                        'srv_diff_host_rate': 0.0,
                        'dst_host_count': np.random.randint(1, 10),
                        'dst_host_srv_count': np.random.randint(1, 10),
                        'dst_host_same_srv_rate': 1.0,
                        'dst_host_diff_srv_rate': 0.0,
                        'dst_host_same_src_port_rate': 0.0,
                        'dst_host_srv_diff_host_rate': 0.0,
                        'dst_host_serror_rate': 0.0,
                        'dst_host_srv_serror_rate': 0.0,
                        'dst_host_rerror_rate': 0.0,
                        'dst_host_srv_rerror_rate': 0.0,
                        'latitude': state[6] if state[6] is not None else np.random.uniform(region['lat_min'], region['lat_max']),
                        'longitude': state[5] if state[5] is not None else np.random.uniform(region['lon_min'], region['lon_max']),
                        'altitude': state[7] if state[7] is not None else np.random.uniform(0, 40000),
                        'velocity': state[9] if state[9] is not None else np.random.uniform(0, 600)
                    })
                log_user_activity("system", f"Fetched {len(adsb_records)} ADS-B records")
                return adsb_records
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded. Retrying after delay...")
                time.sleep(20 * (attempt + 1))
                continue
            else:
                raise Exception(f"Failed to fetch ADS-B data: {response.status_code}")
    except Exception as e:
        logger.error(f"ADS-B fetch error: {str(e)}")
        st.warning(f"ADS-B fetch error: {str(e)}. Using simulated data.")
        return simulate_aviation_traffic(num_samples, region)

def periodic_adsb_fetch(num_samples, region, interval=60):
    while 'adsb_running' in st.session_state and st.session_state.adsb_running:
        traffic_data = fetch_adsb_data(num_samples, region)
        traffic_df = pd.DataFrame(traffic_data)
        traffic_df, anomalies = detect_air_traffic_anomalies(traffic_df)
        results = []
        for _, row in traffic_df.iterrows():
            prediction, confidence = predict_traffic(
                pd.DataFrame([row]), st.session_state.model, st.session_state.scaler,
                st.session_state.label_encoders, st.session_state.le_class
            )
            results.append({
                'timestamp': row['timestamp'],
                'icao24': row['icao24'],
                'airport_code': row['airport_code'],
                'prediction': prediction,
                'confidence': confidence,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'altitude': row['altitude'],
                'velocity': row['velocity'],
                'source': 'ads-b'
            })
        st.session_state.atc_results = results
        st.session_state.atc_anomalies = anomalies
        conflicts = detect_collision_risks(results)
        st.session_state.flight_conflicts = conflicts
        routes = optimize_traffic_flow(results)
        st.session_state.optimized_routes = routes
        if anomalies.empty and not conflicts:
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'ATC Monitoring',
                'severity': 'low',
                'details': f"Fetched {len(traffic_data)} ADS-B records, no anomalies or conflicts"
            })
        time.sleep(interval)

# Simulate aviation traffic
def simulate_aviation_traffic(num_samples=10, region=None):
    try:
        if region is None:
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
        airports = ['DNMM', 'DNAA', 'DNKN', 'DNPO']
        data = {
            'timestamp': pd.date_range(start='now', periods=num_samples, freq='S'),
            'icao24': [f"ICAO{i:06d}" for i in range(num_samples)],
            'protocol_type': np.random.choice(['ads-b', 'acars', 'tcp'], num_samples),
            'service': np.random.choice(['atc', 'flight_data', 'other'], num_samples),
            'src_bytes': np.random.randint(100, 1000, num_samples),
            'dst_bytes': np.random.randint(100, 1000, num_samples),
            'airport_code': np.random.choice(airports, num_samples),
            'duration': np.random.randint(0, 100, num_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ'], num_samples),
            'count': np.random.randint(1, 10, num_samples),
            'srv_count': np.random.randint(1, 10, num_samples),
            'serror_rate': np.random.uniform(0, 1, num_samples),
            'srv_serror_rate': np.random.uniform(0, 1, num_samples),
            'rerror_rate': np.random.uniform(0, 1, num_samples),
            'srv_rerror_rate': np.random.uniform(0, 1, num_samples),
            'same_srv_rate': np.random.uniform(0, 1, num_samples),
            'diff_srv_rate': np.random.uniform(0, 1, num_samples),
            'srv_diff_host_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_count': np.random.randint(1, 10, num_samples),
            'dst_host_srv_count': np.random.randint(1, 10, num_samples),
            'dst_host_same_srv_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_diff_srv_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_same_src_port_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srv_diff_host_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_serror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srv_serror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_rerror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srv_rerror_rate': np.random.uniform(0, 1, num_samples),
            'latitude': np.random.uniform(region['lat_min'], region['lat_max'], num_samples),
            'longitude': np.random.uniform(region['lon_min'], region['lon_max'], num_samples),
            'altitude': np.random.uniform(0, 40000, num_samples),
            'velocity': np.random.uniform(0, 600, num_samples)
        }
        log_user_activity("system", "Simulated ATC traffic")
        return pd.DataFrame(data).to_dict('records')
    except Exception as e:
        logger.error(f"ATC simulation error: {str(e)}")
        st.error(f"ATC simulation error: {str(e)}")
        return []

# Anomaly Detection in Air Traffic Data
def detect_air_traffic_anomalies(df):
    try:
        features = ['latitude', 'longitude', 'altitude', 'velocity']
        X = df[features].fillna(0)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        predictions = model.predict(X)
        df['anomaly'] = predictions == -1
        anomalies = df[df['anomaly']]
        log_user_activity("system", f"Detected {len(anomalies)} air traffic anomalies")
        return df, anomalies
    except Exception as e:
        logger.error(f"Air traffic anomaly detection error: {str(e)}")
        st.error(f"Air traffic anomaly detection error: {str(e)}")
        return df, pd.DataFrame()

# Flight Data Processing: Conflict Prediction
def detect_collision_risks(traffic_data, distance_threshold=5, time_threshold=300):
    try:
        df = pd.DataFrame(traffic_data)
        risks = []
        for i, row1 in df.iterrows():
            for j, row2 in df.iloc[i+1:].iterrows():
                if row1['icao24'] == row2['icao24']:
                    continue
                pos1 = (row1['latitude'], row1['longitude'])
                pos2 = (row2['latitude'], row2['longitude'])
                dist = geodesic(pos1, pos2).km
                vel1 = row1['velocity'] * np.array([np.cos(np.radians(row1.get('heading', 0))), np.sin(np.radians(row1.get('heading', 0)))])
                vel2 = row2['velocity'] * np.array([np.cos(np.radians(row2.get('heading', 0))), np.sin(np.radians(row2.get('heading', 0)))])
                rel_vel = np.linalg.norm(vel1 - vel2)
                time_to_collision = dist / (rel_vel / 3600) if rel_vel > 0 else float('inf')
                if dist < distance_threshold and time_to_collision < time_threshold:
                    risks.append({
                        'icao24_1': row1['icao24'],
                        'icao24_2': row2['icao24'],
                        'distance_km': dist,
                        'time_to_collision_sec': time_to_collision,
                        'severity': 'critical' if time_to_collision < 60 else 'high'
                    })
        if risks:
            for risk in risks:
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Flight Conflict',
                    'severity': risk['severity'],
                    'details': f"Potential collision between {risk['icao24_1']} and {risk['icao24_2']}: {risk['distance_km']:.2f}km, {risk['time_to_collision_sec']:.0f}s"
                })
        log_user_activity("system", f"Detected {len(risks)} flight conflicts")
        return risks
    except Exception as e:
        logger.error(f"Collision risk detection error: {str(e)}")
        st.error(f"Collision risk detection error: {str(e)}")
        return []

# Flight Data Processing: Traffic Flow Optimization
def optimize_traffic_flow(traffic_data, num_clusters=3):
    try:
        df = pd.DataFrame(traffic_data)
        if len(df) < 2:
            return []
        features = ['latitude', 'longitude', 'altitude']
        X = df[features].fillna(0)
        kmeans = KMeans(n_clusters=min(num_clusters, len(X)), random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        routes = []
        for cluster in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster]
            centroid = {
                'latitude': cluster_df['latitude'].mean(),
                'longitude': cluster_df['longitude'].mean(),
                'altitude': cluster_df['altitude'].mean()
            }
            routes.append({
                'cluster': cluster,
                'icao24_list': cluster_df['icao24'].tolist(),
                'suggested_route': centroid,
                'aircraft_count': len(cluster_df)
            })
        log_user_activity("system", f"Optimized traffic flow for {len(routes)} clusters")
        return routes
    except Exception as e:
        logger.error(f"Traffic flow optimization error: {str(e)}")
        st.error(f"Traffic flow optimization error: {str(e)}")
        return []

# Drone Detection
def periodic_drone_detection(interval=120):
    try:
        while 'drone_running' in st.session_state and st.session_state.drone_running:
            # Simulate drone detection in the region
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}  # Default Nigeria region
            num_drones = np.random.randint(0, 5)  # Random number of drones
            drone_results = []
            for i in range(num_drones):
                altitude = np.random.uniform(0, 1000)  # Drones typically fly low
                is_unauthorized = altitude < 400 and np.random.random() > 0.7  # Unauthorized if low and random chance
                drone_results.append({
                    'timestamp': pd.Timestamp.now(),
                    'drone_id': f"DRN{i:03d}",
                    'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
                    'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
                    'altitude': altitude,
                    'status': 'unauthorized' if is_unauthorized else 'authorized',
                    'severity': 'high' if is_unauthorized else 'low'
                })
            st.session_state.drone_results = drone_results
            if any(d['status'] == 'unauthorized' for d in drone_results):
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Drone Intrusion',
                    'severity': 'high',
                    'details': f"Detected {sum(d['status'] == 'unauthorized' for d in drone_results)} unauthorized drones"
                })
            log_user_activity("system", f"Drone detection: {len(drone_results)} drones, {sum(d['status'] == 'unauthorized' for d in drone_results)} unauthorized")
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Drone detection error: {str(e)}")
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Drone Detection',
            'severity': 'error',
            'details': f"Drone detection failed: {str(e)}"
        })

# Threat Intelligence (Transformers Bypassed)
def fetch_threat_feeds(api_key=None):
    try:
        if not api_key:
            logger.warning("No OTX API key provided. Using sample data.")
            return ["Suspicious IP 192.168.1.100 detected", "Malware signature found"]
        url = "https://otx.alienvault.com/api/v1/pulses/subscribed"
        headers = {"X-OTX-API-KEY": api_key}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            pulses = response.json().get('results', [])
            feeds = [pulse['description'] for pulse in pulses if pulse.get('description')]
            return feeds[:10]
        else:
            raise Exception(f"Failed to fetch OTX feeds: {response.status_code}")
    except Exception as e:
        logger.error(f"OTX fetch error: {str(e)}")
        return []

def analyze_threat_feeds(text_data):
    try:
        st.warning("Transformers disabled due to import error. Using regex.")
        threats = [text for text in text_data if re.search(r'suspicious|malware|attack|intrusion', text, re.I)]
        log_user_activity("system", f"Detected {len(threats)} cyber threats")
        return threats
    except Exception as e:
        logger.error(f"Threat intelligence error: {str(e)}")
        st.error(f"Threat intelligence error: {str(e)}")
        return []

def periodic_threat_fetch(api_key, interval=300):
    while 'threat_running' in st.session_state and st.session_state.threat_running:
        feeds = fetch_threat_feeds(api_key)
        threats = analyze_threat_feeds(feeds)
        st.session_state.threats = threats
        if threats:
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Threat Intelligence',
                'severity': 'high',
                'details': f"Detected {len(threats)} cyber threats"
            })
        time.sleep(interval)

# Main application
def main():
    apply_wicket_css()
    setup_user_db()

    if not st.session_state.authenticated:
        st.markdown('<div class="auth-container"><div class="auth-overlay"></div><div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-form"><h2>NAMA IDPS Login</h2>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username", placeholder="Enter username", help="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password", help="Enter your password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login", key="login_btn", help="Click to login"):
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    log_user_activity(username, "Logged in")
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        with col2:
            if st.button("Register", key="register_btn", help="Click to register"):
                if register_user(username, password):
                    st.success("Registration successful! Please login.")
                    log_user_activity(username, "Registered")
                else:
                    st.error("Username already exists or registration failed")
        
        st.markdown('<a href="#" class="auth-link">Forgot Password?</a>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        return

    st.sidebar.image("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/nama_logo.jpg", use_column_width=True)
    
   # Enhanced Sidebar without Icons
menu_options = [
    "Dashboard",
    "Network Scan",
    "ATC Monitoring",
    "Radar Surveillance",
    "Drone Detection",
    "Threat Intelligence",
    "Compliance & Reporting",
    "Settings",
]
menu = st.sidebar.selectbox(
    "Select Module",
    menu_options,
    help="Navigate through IDPS modules"
)

    # Load ML models
    try:
        st.session_state.model = joblib.load('nama_idps_model.pkl')
        st.session_state.scaler = joblib.load('scaler.pkl')
        st.session_state.label_encoders = joblib.load('label_encoders.pkl')
        st.session_state.le_class = joblib.load('le_class.pkl')
    except Exception:
        st.session_state.model = XGBClassifier(random_state=42)
        st.session_state.scaler = StandardScaler()
        st.session_state.label_encoders = {col: LabelEncoder() for col in CATEGORICAL_COLS}
        st.session_state.le_class = LabelEncoder()
        logger.warning("ML models not found. Using defaults.")

    def predict_traffic(df, model, scaler, label_encoders, le_class):
        try:
            df_processed, _, _ = preprocess_data(df, label_encoders, le_class, is_train=False)
            if df_processed is None:
                return "Error", 0.0
            features = [col for col in NSL_KDD_COLUMNS if col not in ['class'] + LOW_IMPORTANCE_FEATURES]
            features = [col for col in features if col in df_processed.columns]
            X = df_processed[features]
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)
            confidence = model.predict_proba(X_scaled).max(axis=1)[0]
            prediction_label = le_class.inverse_transform(prediction)[0]
            return prediction_label, confidence
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            st.error(f"Prediction error: {str(e)}")
            return "Error", 0.0

    if menu == "Dashboard":
        st.header("NAMA IDPS Dashboard")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("System Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detection Rate", f"{st.session_state.compliance_metrics['detection_rate']}%", help="Percentage of threats detected")
        with col2:
            st.metric("Open Ports", st.session_state.compliance_metrics['open_ports'], help="Number of open network ports")
        with col3:
            st.metric("Active Alerts", st.session_state.compliance_metrics['alerts'], help="Current active alerts")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent Alerts")
        if st.session_state.alert_log:
            alert_df = pd.DataFrame(st.session_state.alert_log)
            st.dataframe(
                alert_df[['timestamp', 'type', 'severity', 'details']],
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "severity": st.column_config.TextColumn("Severity", help="Alert severity level")
                }
            )
        else:
            st.write("No alerts yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "Network Scan":
        st.header("Real-Time Network Vulnerability Scan")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        scan_type = st.selectbox("Scan Type", ["TCP SYN", "TCP Connect", "UDP"], help="Select NMAP scan type")
        target = st.text_input("Target IP/Hostname", "192.168.1.1", help="Enter target IP or hostname")
        port_range = st.text_input("Port Range", "1-1000", help="Specify port range (e.g., 1-1000)")
        custom_args = st.text_input("Custom NMAP Arguments", "-Pn", help="Additional NMAP arguments")
        scan_interval = st.slider("Scan Interval (seconds)", 60, 600, 300, help="Frequency of scans")
        
        if 'nmap_running' not in st.session_state:
            st.session_state.nmap_running = False
            st.session_state.scan_results = []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Real-Time Scan", key="start_nmap"):
                st.session_state.nmap_running = True
                threading.Thread(
                    target=periodic_nmap_scan,
                    args=(target, scan_type, port_range, custom_args, scan_interval),
                    daemon=True
                ).start()
                st.success("Real-time scanning started!")
        with col2:
            if st.button("Stop Scan", key="stop_nmap"):
                st.session_state.nmap_running = False
                st.success("Real-time scanning stopped.")
        
        if st.session_state.scan_results:
            st.subheader("Latest Scan Results")
            st.dataframe(
                pd.DataFrame(st.session_state.scan_results),
                use_container_width=True,
                column_config={
                    "port": st.column_config.NumberColumn("Port"),
                    "state": st.column_config.TextColumn("State", help="Port status (open/closed)")
                }
            )
            open_ports = len([r for r in st.session_state.scan_results if r['state'] == 'open'])
            st.session_state.compliance_metrics['open_ports'] = open_ports
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Network Scan',
                'severity': 'medium',
                'details': f"Scanned {target}, found {open_ports} open ports"
            })
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "ATC Monitoring":
        st.header("Real-Time Air Traffic Control Monitoring")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        data_source = st.radio("Data Source", ["Simulated", "Real ADS-B"], help="Choose data source")
        num_samples = st.slider("Number of Samples", 1, 100, 10, help="Number of aircraft to display")
        fetch_interval = st.slider("Fetch Interval (seconds)", 30, 300, 60, help="Data refresh rate")
        region_type = st.selectbox("Region Type", ["All Nigeria", "Remote Areas", "Urban Areas"], help="Select monitoring region")
        regions = {
            "All Nigeria": {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15},
            "Remote Areas": {'lat_min': 10, 'lat_max': 14, 'lon_min': 10, 'lon_max': 15},
            "Urban Areas": {'lat_min': 6, 'lat_max': 9, 'lon_min': 3, 'lon_max': 8}
        }
        region = regions[region_type]
        
        if 'adsb_running' not in st.session_state:
            st.session_state.adsb_running = False
            st.session_state.atc_results = []
            st.session_state.atc_anomalies = pd.DataFrame()
            st.session_state.flight_conflicts = []
            st.session_state.optimized_routes = []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Real-Time Monitoring", key="start_atc"):
                st.session_state.adsb_running = True
                if data_source == "Simulated":
                    st.session_state.atc_results = simulate_aviation_traffic(num_samples, region)
                    st.session_state.atc_anomalies = pd.DataFrame()
                    st.session_state.flight_conflicts = []
                    st.session_state.optimized_routes = []
                else:
                    threading.Thread(
                        target=periodic_adsb_fetch,
                        args=(num_samples, region, fetch_interval),
                        daemon=True
                    ).start()
                st.success("Real-time ATC monitoring started!")
        with col2:
            if st.button("Stop Monitoring", key="stop_atc"):
                st.session_state.adsb_running = False
                st.success("Real-time ATC monitoring stopped.")
        
        if st.session_state.atc_results:
            st.subheader("Latest ATC Data")
            results_df = pd.DataFrame(st.session_state.atc_results)
            st.dataframe(
                results_df[['timestamp', 'icao24', 'airport_code', 'prediction', 'confidence', 'latitude', 'longitude', 'altitude']],
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")
                }
            )
            
            if not st.session_state.atc_anomalies.empty:
                st.subheader("Detected Anomalies")
                st.dataframe(
                    st.session_state.atc_anomalies[['timestamp', 'icao24', 'airport_code', 'latitude', 'longitude', 'altitude']],
                    use_container_width=True
                )
            
            if st.session_state.flight_conflicts:
                st.subheader("Flight Conflicts")
                st.dataframe(
                    pd.DataFrame(st.session_state.flight_conflicts),
                    use_container_width=True,
                    column_config={
                        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
                        "time_to_collision_sec": st.column_config.NumberColumn("Time to Collision (s)", format="%.0f")
                    }
                )
            
            if st.session_state.optimized_routes:
                st.subheader("Optimized Traffic Routes")
                routes_df = pd.DataFrame([
                    {'cluster': r['cluster'], 'aircraft_count': r['aircraft_count'],
                     'suggested_latitude': r['suggested_route']['latitude'],
                     'suggested_longitude': r['suggested_route']['longitude'],
                     'suggested_altitude': r['suggested_route']['altitude']}
                    for r in st.session_state.optimized_routes
                ])
                st.dataframe(routes_df, use_container_width=True)
                fig_routes = px.scatter(
                    results_df,
                    x='longitude',
                    y='latitude',
                    color='airport_code',
                    size='altitude',
                    title="Optimized Traffic Flow",
                    hover_data=['icao24'],
                    template='plotly_dark'
                )
                for route in st.session_state.optimized_routes:
                    fig_routes.add_trace(go.Scatter(
                        x=[route['suggested_route']['longitude']],
                        y=[route['suggested_route']['latitude']],
                        mode='markers',
                        marker=dict(size=15, symbol='star', color=WICKET_THEME['error']),
                        name=f"Cluster {route['cluster']} Centroid"
                    ))
                st.plotly_chart(fig_routes, use_container_width=True)
            
            fig = px.scatter(
                results_df,
                x='longitude',
                y='latitude',
                size='altitude',
                color='prediction',
                hover_data=['icao24', 'velocity'],
                title="Real-Time Air Traffic Visualization",
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "Radar Surveillance":
        st.header("Real-Time Radar Surveillance")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        num_targets = st.slider("Number of Radar Targets", 1, 20, 5, help="Number of radar targets to simulate")
        fetch_interval = st.slider("Update Interval (seconds)", 30, 300, 60, help="Radar refresh rate")
        region_type = st.selectbox("Region Type", ["All Nigeria", "Remote Areas", "Urban Areas"], help="Select radar region")
        regions = {
            "All Nigeria": {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15},
            "Remote Areas": {'lat_min': 10, 'lat_max': 14, 'lon_min': 10, 'lon_max': 15},
            "Urban Areas": {'lat_min': 6, 'lat_max': 9, 'lon_min': 3, 'lon_max': 8}
        }
        region = regions[region_type]
        
        if 'radar_running' not in st.session_state:
            st.session_state.radar_running = False
            st.session_state.radar_data = []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Radar Surveillance", key="start_radar"):
                st.session_state.radar_running = True
                threading.Thread(
                    target=periodic_radar_update,
                    args=(num_targets, region, fetch_interval),
                    daemon=True
                ).start()
                st.success("Radar surveillance started!")
        with col2:
            if st.button("Stop Radar Surveillance", key="stop_radar"):
                st.session_state.radar_running = False
                st.success("Radar surveillance stopped!")
        
        if st.session_state.radar_data:
            st.subheader("Radar and ADS-B Tracking")
            fig = display_radar(st.session_state.radar_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                pd.DataFrame(st.session_state.radar_data)[['timestamp', 'target_id', 'source', 'latitude', 'longitude', 'altitude', 'velocity', 'heading']],
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                    "longitude": st.column_config.NumberColumn("Longitude", format="%.4f")
                }
            )
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "Drone Detection":
        st.header("Real-Time Drone Intrusion Detection")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        detection_interval = st.slider("Detection Interval (seconds)", 60, 300, 120, help="Drone scan frequency")
        
        if 'drone_running' not in st.session_state:
            st.session_state.drone_running = False
            st.session_state.drone_results = []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Drone Detection", key="start_drone"):
                st.session_state.drone_running = True
                threading.Thread(
                    target=periodic_drone_detection,
                    args=(detection_interval,),
                    daemon=True
                ).start()
                st.success("Real-time drone detection started!")
        with col2:
            if st.button("Stop Drone Detection", key="stop_drone"):
                st.session_state.drone_running = False
                st.success("Drone detection completed successfully.")
            
        if st.session_state.drone_results:
            st.subheader("Detected Drone Intrusions")
            st.dataframe(
                pd.DataFrame(st.session_state.drone_results),
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "altitude": st.column_config["altitude"]("Altitude", format="%.0f"),
                    "status": st.column_config.TextColumn("Status", help="Drone authorization status")
                })
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "Threat Intelligence":
        st.header("Real-Time Cyber Threat Intelligence")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        otx_api_key = st.text_input("AlienVault OTX API Key (optional)", type="password", help="Enter your OTX API key (leave blank for sample data)")
        threat_feed = st.text_area("Manual Threat Feed (one per line)", "Suspicious IP detected\nMalware signature found", help="Enter manual threat feeds")
        fetch_interval = st.slider("Fetch Interval (seconds)", 60, 600, 300, help="Threat feed refresh rate")
        
        if 'threat_running' not in st.session_state:
            st.session_state.threat_running = False
            st.session_state.threats = []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Real-Time Threat Analysis", key="start_threat"):
                st.session_state.threat_running = True
                if threat_feed:
                    threats = analyze_threat_feeds(threat_feed.split('\n'))
                    st.session_state.threats = threats
                threading.Thread(
                    target=periodic_threat_fetch,
                    args=(otx_api_key, fetch_interval),
                    daemon=True
                ).start()
                st.success("Real-time threat analysis started!")
        with col2:
            if st.button("Stop Threat Analysis", key="stop_threat"):
                st.session_state.threat_running = False
                st.success("Real-time threat analysis stopped.")
        
        if st.session_state.threats:
            st.subheader("Detected Threats")
            for threat in st.session_state.threats:
                st.write(f"- {threat}")
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "Compliance & Reporting":
        st.header("Compliance & Reporting")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Compliance Metrics")
        st.write(st.session_state.compliance_metrics)
        
        if st.button("Generate Report", key="generate_report", help="Download compliance report"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = doc.getSampleStyleSheet()
            elements = []
            
            elements.append(Paragraph("NAMA IDPS Report", styles['Title']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Generated on: {datetime.now()}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            if st.session_state.alert_log:
                elements.append(Paragraph("Recent Alerts", styles['Heading2']))
                alert_data = [[a['timestamp'], a['type'], a['severity'], a['details']] for a in st.session_state.alert_log]
                alert_table = Table([[['Timestamp', 'Type', 'Severity', 'Details']]] + alert_data)
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
            
            doc.build(elements)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64, {b64}" download="nama_idps_report.pdf">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "Settings":
        st.header("System Settings")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Configure system settings here.")
        st.session_state.theme_mode = st.selectbox("Theme Mode", ["Dark", "Light"], index=0 if st.session_state.theme_mode == 'dark' else 1, help="Select theme")
        if st.button("Logout", key="logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

