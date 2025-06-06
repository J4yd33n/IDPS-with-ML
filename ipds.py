
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
import streamlit.components.v1 as components

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
                background: url('https://github.com/J4yd33n/IDPS-with-ML/blob/main/airplane.jpg') no-repeat center center fixed;
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
                position: relative;
                overflow: hidden;
            }}
            
            @keyframes slideInAuth {{
                0% {{ transform: translateX(100vw); opacity: 0; }}
                100% {{ transform: translateX(0); opacity: 1; }}
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
                position: relative;
                overflow: hidden;
            }}
            .auth-btn:hover {{
                background: {WICKET_THEME['hover']};
                box-shadow: 0 0 25px {WICKET_THEME['hover']};
                transform: scale(1.05);
            }}
            .auth-btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(
                    90deg,
                    transparent,
                    rgba(255, 255, 255, 0.3),
                    transparent
                );
                transition: 0.5s;
            }}
            .auth-btn:hover::before {{
                left: 100%;
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
            
            .auth-neon-text {{
                font-family: 'Orbitron', sans-serif;
                color: {WICKET_THEME['accent']};
                text-shadow: 
                    0 0 5px {WICKET_THEME['accent']},
                    0 0 10px {WICKET_THEME['accent']},
                    0 0 20px {WICKET_THEME['accent']},
                    0 0 40px {WICKET_THEME['hover']};
                animation: neonFlicker 1.5s infinite alternate;
            }}
            
            @keyframes neonFlicker {{
                0% {{ text-shadow: 
                    0 0 5px {WICKET_THEME['accent']},
                    0 0 10px {WICKET_THEME['accent']},
                    0 0 20px {WICKET_THEME['accent']},
                    0 0 40px {WICKET_THEME['hover']}; }}
                100% {{ text-shadow: 
                    0 0 10px {WICKET_THEME['accent']},
                    0 0 20px {WICKET_THEME['accent']},
                    0 0 30px {WICKET_THEME['accent']},
                    0 0 50px {WICKET_THEME['hover']}; }}
            }}
            
            .auth-particle {{
                position: absolute;
                background: {WICKET_THEME['accent']};
                border-radius: 50%;
                pointer-events: none;
                animation: float 6s infinite linear;
                z-index: 1;
            }}
            
            @keyframes float {{
                0% {{ transform: translateY(100vh); opacity: 0.6; }}
                100% {{ transform: translateY(-100vh); opacity: 0; }}
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
                .auth-card {{
                    padding: 1.5rem;
                    max-width: 90%;
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
if 'satellite_results' not in st.session_state:
    st.session_state.satellite_results = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'le_class' not in st.session_state:
    st.session_state.le_class = None

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
            adsb_df['heading'] = adsb_df.get('heading', np.random.uniform(0, 360))
        combined = pd.concat([radar_df, adsb_df], ignore_index=True)
        return combined.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to merge Radar-ADS-B data: {str(e)}")
        st.error(f"Failed to merge Radar-ADS-B data: {str(e)}")
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
                    symbol='cross',
                    line=dict(width=2, color=WICKET_THEME['text']),
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

def periodic_radar_update(num_targets, interval=60):
    while 'radar_running' in st.session_state and st.session_state.radar_running:
        try:
            radar_data = simulate_radar_data(num_targets)
            adsb_data = st.session_state.get('atc_results', [])
            combined_data = merge_radar_adsb(radar_data, adsb_data)
            st.session_state.radar_data = combined_data
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Periodic radar update error: {str(e)}")
            st.error(f"Error: {str(e)}")
            continue

# Real-time NMAP scan
def run_nmap_scan(target, scan_type, port_range, custom_args=None):
    try:
        if not NMAP_AVAILABLE:
            raise ImportError("python-nmap library is not installed")
        nm = nmap.PortScanner()
        scan_args = {
            'TCP SYN': '-sS',
            'TCP Connect': '-sT',
            'UDP': '-sU'
        }.get(scan_type, '-sS')
        args = f"{scan_args} {custom_args}" if custom_args else scan_args
        nm.scan(target, port_range, arguments=args)
        results = []
        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                ports = nm[host][proto].keys()
                for port in ports:
                    state = nm[host][proto][port]['state']
                    service = nm[host][proto][port].get('name', 'unknown')
                    results.append({
                        'port': port,
                        'protocol': proto,
                        'state': state,
                        'service': service
                    })
        log_user_activity("system", f"Performed NMAP scan on {target}")
        return results
    except Exception as e:
        logger.error(f"NMAP scan error: {str(e)}")
        st.warning(f"NMAP scan error: {str(e)}. Using simulated data.")
        return simulate_nmap_scan(target, scan_type, port_range)

# Periodic NMAP scan
def periodic_nmap_scan(target, scan_type, port_range, custom_args=None, interval=3600):
    while 'nmap_running' in st.session_state and st.session_state.nmap_running:
        try:
            results = run_nmap_scan(target, scan_type, port_range, custom_args)
            st.session_state.scan_results = results
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Periodic NMAP scan error: {str(e)}")
            continue

# Simulated NMAP scan
def simulate_nmap_scan(target, scan_type, port_range):
    try:
        common_ports = {
            '21': {'service': 'ftp', 'protocol': 'tcp'},
            '22': {'service': 'ssh', 'protocol': 'tcp'},
            '23': {'service': 'telnet', 'protocol': 'tcp'},
            '80': {'service': 'http', 'protocol': 'tcp'},
            '443': {'service': 'https', 'protocol': 'tcp'},
            '3306': {'service': 'mysql', 'protocol': 'tcp'},
            '3389': {'service': 'rdp', 'protocol': 'tcp'}
        }
        start_port, end_port = map(int, port_range.split('-'))
        ports_to_scan = [p for p in common_ports.keys() if start_port <= int(p) <= end_port]
        np.random.seed(42)
        results = []
        for port in ports_to_scan:
            port_data = common_ports[port]
            protocol = port_data['protocol']
            service = port_data['service']
            if scan_type in ['TCP SYN', 'TCP Connect'] and protocol != 'tcp':
                continue
            if scan_type == 'UDP' and protocol != 'udp':
                continue
            state = 'open' if np.random.random() < 0.5 else 'closed'
            results.append({
                'port': port,
                'protocol': protocol,
                'state': state,
                'service': service
            })
        log_user_activity("system", f"Simulated NMAP scan on {target} for {port_range} with {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Failed to simulate NMAP scan: {str(e)}")
        st.error(f"Error: {str(e)}")
        return []

@st.cache_data
def fetch_adsb_data(num_samples=10, region=None):
    try:
        if region is None:
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
        username = st.secrets.get('OPENSKY_USERNAME', os.getenv('OPENSKY_USERNAME'))
        password = st.secrets.get('OPENSKY_PASSWORD', os.getenv('OPENSKY_PASSWORD'))
        params = {
            'lamin': region['lat_min'],
            'lomin': region['lon_min'],
            'lamax': region['lat_max'],
            'lomax': region['lon_max']
        }
        url = "https://opensky-network.org/api/states/all"
        for attempt in range(3):
            try:
                response = requests.get(url, auth=(username, password) if username and password else None, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json().get('states', [])
                    if not data:
                        logger.warning("No ADS-B data found. Using simulated data.")
                        return simulate_radar_data(num_samples, region)
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
                            'dst_bytes': np.random.randint(0, 100),
                            'airport_code': np.random.choice(airports),
                            'duration': np.random.randint(0, 100),
                            'flag': 'SF',
                            'count': np.random.randint(1, 10),
                            'srv_count': np.random.randint(1, 10),
                            'serror_rate': 0.0,
                            'srv_serror_rate': 0.0,
                            'rerror_rate': np.random.uniform(0, 0.1),
                            'same_srv_rate': np.random.uniform(0.8, 1.0),
                            'diff_srv_rate': np.random.uniform(0, 0.2),
                            'srv_diff_host_rate': np.random.uniform(0, 0.1),
                            'dst_host_count': np.random.randint(1, 255),
                            'dst_host_srv_count': np.random.randint(1, 255),
                            'dst_host_same_srv_rate': np.random.uniform(0, 1.0),
                            'dst_host_diff_srv_rate': np.random.uniform(0, 0.2),
                            'dst_host_same_src_port_rate': np.random.uniform(0, 0.5),
                            'dst_host_srv_diff_host_rate': np.random.uniform(0, 0.1),
                            'dst_host_serror_rate': np.random.uniform(0, 0.1),
                            'dst_host_rerror_rate': np.random.uniform(0, 0.1),
                            'dst_host_srv_rerror_rate': np.random.uniform(0, 0.1),
                            'latitude': state[6] if state[6] is not None else np.random.uniform(region['lat_min'], region['lat_max']),
                            'longitude': state[5] if state[5] is not None else np.random.uniform(region['lon_min'], region['lon_max']),
                            'altitude': state[7] if state[7] is not None else np.random.uniform(0, 40000),
                            'velocity': state[9] if state[9] is not None else np.random.uniform(0, 600)
                        })
                    log_user_activity("system", f"Fetched {len(adsb_records)} ADS-B data records")
                    return adsb_records
                elif response.status_code == 429:
                    logger.warning("OpenSky API rate limit exceeded. Retrying...")
                    time.sleep(30 * (attempt + 1))
                    continue
                else:
                    raise Exception(f"Failed to fetch ADS-B: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching ADS-B data: {str(e)}")
                if attempt == 2:
                    logger.warning("Max retries reached. Using simulated data.")
                    return simulate_radar_data(num_samples, region)
    except Exception as e:
        logger.error(f"Error in fetch_adsb_data: {str(e)}")
        return simulate_radar_data(num_samples, region)

def simulate_aviation_traffic(num_samples=10, region=None):
    try:
        if region is None:
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
        airports = ['DNMM', 'DNAA', 'DNKN', 'DNPO']
        data = []
        for i in range(num_samples):
            data.append({
                'timestamp': pd.Timestamp.now(),
                'icao24': f"ICAO{i:04d}",
                'protocol_type': np.random.choice(['ads-b', 'acars', 'mode-s']),
                'service': np.random.choice(['flight_data', 'atc', 'other']),
                'src_bytes': np.random.randint(100, 1000),
                'dst_bytes': np.random.randint(0, 100),
                'airport_code': np.random.choice(airports),
                'duration': np.random.randint(0, 100),
                'flag': np.random.choice(['SF', 'S0', 'REJ']),
                'count': np.random.randint(1, 10),
                'srv_count': np.random.randint(1, 10),
                'serror_rate': np.random.uniform(0, 0.1),
                'srv_serror_rate': np.random.uniform(0, 0.1),
                'rerror_rate': np.random.uniform(0, 0.1),
                'same_srv_rate': np.random.uniform(0.8, 1.0),
                'diff_srv_rate': np.random.uniform(0, 0.2),
                'srv_diff_host_rate': np.random.uniform(0, 0.1),
                'dst_host_count': np.random.randint(1, 255),
                'dst_host_srv_count': np.random.randint(1, 255),
                'dst_host_same_srv_rate': np.random.uniform(0, 1.0),
                'dst_host_diff_srv_rate': np.random.uniform(0, 0.2),
                'dst_host_same_src_port_rate': np.random.uniform(0, 0.5),
                'dst_host_srv_diff_host_rate': np.random.uniform(0, 0.1),
                'dst_host_serror_rate': np.random.uniform(0, 0.1),
                'dst_host_rerror_rate': np.random.uniform(0, 0.1),
                'dst_host_srv_rerror_rate': np.random.uniform(0, 0.1),
                'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
                'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
                'altitude': np.random.uniform(0, 40000),
                'velocity': np.random.uniform(0, 600)
            })
        log_user_activity("system", f"Simulated {num_samples} aviation traffic records")
        return data
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        return []

def periodic_adsb_fetch(num_samples, region=None, interval=10):
    try:
        while 'adsb_running' in st.session_state and st.session_state.get('adsb_running', False):
            try:
                traffic_data = fetch_adsb_data(num_samples, region)
                traffic_df = pd.DataFrame(traffic_data)
                traffic_df, anomalies = detect_aircraft_anomalies(traffic_df)
                results = []
                for _, row in traffic_df.iterrows():
                    try:
                        prediction, confidence = predict_traffic_data(
                            pd.DataFrame([row]),
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.label_encoders,
                            st.session_state.le_class
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
                    except Exception as e:
                        logger.error(f"Prediction error: {str(e)}")
                        continue
                st.session_state.atc_results = results
                st.session_state.atc_anomalies = anomalies
                conflicts = detect_collision_risks(results)
                st.session_state.flight_conflicts = conflicts
                st.session_state.optimized_routes = optimize_routes(results)
                if not anomalies.empty or conflicts:
                    st.session_state.alert_log.append({
                        'timestamp': datetime.now(),
                        'type': 'ATC Monitoring',
                        'severity': 'high',
                        'details': f"Detected {len(anomalies)} anomalies and {len(conflicts)} conflicts"
                    })
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Periodic ADS-B fetch error: {str(e)}")
                st.error(f"Periodic ADS-B fetch error: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error in periodic_adsb_fetch: {str(e)}")
        st.error(f"Error in periodic_adsb_fetch: {str(e)}")

def detect_drones():
    try:
        region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
        num_drones = np.random.randint(0, 5)
        drone_results = []
        for i in range(num_drones):
            altitude = np.random.uniform(0, 1000)
            is_unauthorized = altitude < 400
            drone_results.append({
                'timestamp': pd.Timestamp.now(),
                'drone_id': f"DRN{i:03d}",
                'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
                'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
                'altitude': altitude,
                'status': 'unidentified' if is_unauthorized else 'authorized',
                'severity': 'high' if is_unauthorized else 'low'
            })
        st.session_state.drone_results = drone_results
        if any(d['status'] == 'unidentified' for d in drone_results):
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Drone Intrusion',
                'severity': 'high',
                'details': f"Detected {sum(d['status'] == 'unidentified' for d in drone_results)} unauthorized drones"
            })
        log_user_activity("system", f"Detected {len(drone_results)} drones, {sum(d['status'] == 'unidentified' for d in drone_results)} unauthorized")
    except Exception as e:
        logger.error(f"Drone detection error: {str(e)}")
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Drone Detection Error',
            'severity': 'high',
            'details': f"Drone detection failed: {str(e)}"
        })

def periodic_drone_detection(interval=3600):
    while 'drone_running' in st.session_state and st.session_state.drone_running:
        detect_drones()
        time.sleep(interval)

def detect_collision_risks(data):
    try:
        df = pd.DataFrame(data)
        risks = []
        for i, row1 in df.iterrows():
            for j, row2 in df.iloc[i+1:].iterrows():
                if row1['icao24'] == row2['icao24']:
                    continue
                pos1 = (row1['latitude'], row1['longitude'])
                pos2 = (row2['latitude'], row2['longitude'])
                dist = geodesic(pos1, pos2).km
                if dist < 5:
                    risks.append({
                        'icao24_1': row1['icao24'],
                        'icao24_2': row2['icao24'],
                        'distance_km': dist,
                        'severity': 'critical' if dist < 2 else 'high'
                    })
                    st.session_state.alert_log.append({
                        'timestamp': datetime.now(),
                        'type': 'Flight Conflict',
                        'severity': 'high',
                        'details': f"Collision risk between {row1['icao24']} and {row2['icao24']}: {dist:.2f}km"
                    })
        log_user_activity("system", f"Detected {len(risks)} collision risks")
        return risks
    except Exception as e:
        logger.error(f"Collision detection error: {str(e)}")
        return []

def detect_aircraft_anomalies(df):
    try:
        features = ['latitude', 'longitude', 'altitude', 'velocity']
        X = df[features].fillna(0)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        predictions = model.predict(X)
        df['anomaly'] = predictions == -1
        anomalies = df[df['anomaly']]
        return df, anomalies
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        return df, pd.DataFrame()

def optimize_routes(data):
    try:
        df = pd.DataFrame(data)
        if df.empty:
            return []
        routes = []
        for _, row in df.iterrows():
            current_pos = (row['latitude'], row['longitude'])
            velocity = row['velocity']
            new_lat = current_pos[0] + (velocity * np.cos(np.radians(0)) / 3600)
            new_lon = current_pos[1] + (velocity * np.sin(np.radians(0)) / 3600)
            routes.append({
                'icao24': row['icao24'],
                'current_pos': current_pos,
                'new_pos': (new_lat, new_lon),
                'velocity': velocity
            })
        return routes
    except Exception as e:
        logger.error(f"Route optimization error: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def fetch_threat_intelligence(api_key=None, source="alienvault"):
    try:
        if source == "alienvault" and api_key:
            url = "https://otx.alienvault.com/api/v1/pulses/subscribed"
            headers = {'X-OTX-API-KEY': api_key}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get("results", [])
                threats = []
                for pulse in data[:10]:
                    threats.append({
                        'timestamp': pd.Timestamp.now(),
                        'threat_id': pulse.get('id', 'unknown'),
                        'description': pulse.get('name', 'No description'),
                        'indicators': [i.get('indicator', '') for i in pulse.get('indicators', [])],
                        'severity': 'medium',
                        'source': 'alienvault'
                    })
                log_user_activity("system", f"Fetched {len(threats)} threats from AlienVault")
                return threats
        return simulate_threat_intelligence()
    except Exception as e:
        logger.error(f"Threat intelligence fetch error: {str(e)}")
        return simulate_threat_intelligence()

def simulate_threat_intelligence(num_threats=5):
    try:
        threats = []
        for i in range(num_threats):
            threats.append({
                'timestamp': pd.Timestamp.now(),
                'threat_id': f"THR{i:03d}",
                'description': f"Simulated threat {i+1}",
                'indicators': [f"192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}"],
                'severity': np.random.choice(['low', 'medium', 'high']),
                'source': 'simulated'
            })
        return threats
    except Exception as e:
        logger.error(f"Threat simulation error: {str(e)}")
        return []

def periodic_threat_fetch(api_key, interval=3600):
    try:
        while 'threat_running' in st.session_state and st.session_state.threat_running:
            threats = fetch_threat_intelligence(api_key)
            st.session_state.threats = threats
            if any(t['severity'] in ['high', 'medium'] for t in threats):
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Threat Intelligence',
                    'severity': 'high',
                    'details': f"Detected {sum(t['severity'] in ['high', 'medium'] for t in threats)} notable threats"
                })
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Periodic threat fetch error: {str(e)}")
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Threat Fetch Error',
            'severity': 'high',
            'details': f"Threat fetch failed: {str(e)}"
        })

def train_model(df, model_type="xgboost"):
    try:
        label_encoders = {}
        le_class = LabelEncoder()
        df_processed, label_encoders, le_class = preprocess_data(df, label_encoders, le_class, is_train=True)
        if df_processed is None:
            return None, None, None, None
        X = df_processed.drop(columns=['class'], errors='ignore')
        y = df_processed['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if model_type == "xgboost":
            model = XGBClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"Trained {model_type} model with accuracy: {accuracy:.2f}")
        return model, scaler, label_encoders, le_class
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        return None, None, None, None

def predict_traffic_data(df, model, scaler, label_encoders, le_class):
    try:
        df_processed, _, _ = preprocess_data(df, label_encoders, le_class, is_train=False)
        if df_processed is None:
            return "Error", 0
        X = df_processed.drop(columns=['class'], errors='ignore')
        X_scaled = scaler.transform(X)
        probabilities = model.predict_proba(X_scaled)
        prediction = le_class.inverse_transform([np.argmax(probabilities[0])])[0]
        confidence = np.max(probabilities[0])
        if confidence < 0.7:
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Low Confidence Prediction',
                'severity': 'medium',
                'details': f"Prediction {prediction} with confidence {confidence:.2f}"
            })
        return prediction, confidence
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error", 0

def generate_pdf_report(data, filename="nama_idps_report.pdf"):
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        elements.append(Paragraph("NAMA IDPS Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Summary", styles['Heading2']))
        summary_data = [
            ["Report Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Total Alerts", len(st.session_state.alert_log)],
            ["High Severity Alerts", sum(a['severity'] == 'high' for a in st.session_state.alert_log)],
            ["Flight Conflicts", len(st.session_state.flight_conflicts)],
            ["Unauthorized Drones", sum(d['status'] == 'unidentified' for d in st.session_state.drone_results)]
        ]
        table = Table(summary_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Recent Alerts", styles['Heading2']))
        alert_data = [["Timestamp", "Type", "Severity", "Details"]]
        for alert in st.session_state.alert_log[:5]:
            alert_data.append([
                alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                alert['type'],
                alert['severity'],
                alert['details']
            ])
        alert_table = Table(alert_data)
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(alert_table)
        doc.build(elements)
        return filename
    except Exception as e:
        logger.error(f"PDF report generation error: {str(e)}")
        return None

# Compliance Monitoring
def monitor_compliance():
    try:
        compliance_metrics = st.session_state.compliance_metrics
        open_ports = len([r for r in st.session_state.scan_results if r['state'] == 'open']) if st.session_state.scan_results else 0
        alerts_count = len(st.session_state.alert_log)
        detection_rate = (len(st.session_state.atc_anomalies) / len(st.session_state.atc_results) * 100) if st.session_state.atc_results else 0
        
        compliance_metrics.update({
            'detection_rate': detection_rate,
            'open_ports': open_ports,
            'alerts': alerts_count
        })
        
        if open_ports > 10 or alerts_count > 5:
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Compliance Alert',
                'severity': 'high',
                'details': f"High open ports ({open_ports}) or alerts ({alerts_count}) detected"
            })
        
        log_user_activity("system", "Compliance metrics updated")
        return compliance_metrics
    except Exception as e:
        logger.error(f"Compliance monitoring error: {str(e)}")
        st.error(f"Compliance monitoring error: {str(e)}")
        return st.session_state.compliance_metrics

def periodic_compliance_check(interval=3600):
    while 'compliance_running' in st.session_state and st.session_state.compliance_running:
        try:
            metrics = monitor_compliance()
            st.session_state.compliance_metrics = metrics
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Periodic compliance check error: {str(e)}")
            continue

# Display Compliance Metrics
def display_compliance_metrics():
    try:
        metrics = st.session_state.compliance_metrics
        st.markdown("### Compliance Dashboard")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detection Rate", f"{metrics['detection_rate']:.1f}%")
        with col2:
            st.metric("Open Ports", metrics['open_ports'])
        with col3:
            st.metric("Active Alerts", metrics['alerts'])
        
        # Compliance Status Visualization
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
            font={'color': WICKET_THEME['text_light']},
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Compliance Alerts
        st.markdown("#### Recent Compliance Alerts")
        compliance_alerts = [a for a in st.session_state.alert_log if a['type'] == 'Compliance Alert']
        if compliance_alerts:
            st.dataframe(pd.DataFrame(compliance_alerts[-5:]))
        else:
            st.info("No compliance alerts to display.")
    except Exception as e:
        logger.error(f"Compliance display error: {str(e)}")
        st.error(f"Compliance display error: {str(e)}")

# Download Report
def download_report():
    try:
        report_file = generate_pdf_report(st.session_state.alert_log, filename="nama_idps_report.pdf")
        if report_file and os.path.exists(report_file):
            with open(report_file, "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name=report_file,
                    mime="application/pdf"
                )
            os.remove(report_file)
        else:
            st.error("Failed to generate PDF report")
    except Exception as e:
        logger.error(f"Report download error: {str(e)}")
        st.error(f"Report download error: {str(e)}")

# Enhanced Drone Visualization
def display_drone_data():
    try:
        if not st.session_state.drone_results:
            st.warning("No drone data available")
            return
        
        df = pd.DataFrame(st.session_state.drone_results)
        fig = go.Figure()
        
        # Drone positions
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
            margin=dict(l=10, r=10, t=50, b=10),
            title=dict(
                text="Drone Surveillance",
                font=dict(color=WICKET_THEME['text_light'], size=20),
                x=0.5,
                xanchor='center'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[['timestamp', 'drone_id', 'latitude', 'longitude', 'altitude', 'status', 'severity']])
    except Exception as e:
        logger.error(f"Drone display error: {str(e)}")
        st.error(f"Drone display error: {str(e)}")

# Threat Intelligence Visualization
def display_threat_intelligence():
    try:
        if not st.session_state.threats:
            st.warning("No threat intelligence data available")
            return
        
        df = pd.DataFrame(st.session_state.threats)
        st.markdown("### Threat Intelligence Feed")
        
        # Severity Distribution
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
            font={'color': WICKET_THEME['text_light']},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent Threats
        st.markdown("#### Recent Threats")
        st.dataframe(df[['timestamp', 'threat_id', 'description', 'severity', 'source']])
    except Exception as e:
        logger.error(f"Threat intelligence display error: {str(e)}")
        st.error(f"Threat intelligence error: {str(e)}")

def main():
    apply_wicket_css()
    setup_user_db()

    if 'form_type' not in st.session_state:
        st.session_state.form_type = 'signin'

    if not st.session_state.model:
        # Create a complete sample DataFrame with all NSL-KDD columns
        sample_data = pd.DataFrame({
            'duration': [0, 100, 200],
            'protocol_type': ['tcp', 'udp', 'icmp'],
            'service': ['http', 'ftp', 'ssh'],
            'flag': ['SF', 'S0', 'REJ'],
            'src_bytes': [100, 500, 1500],
            'dst_bytes': [0, 100, 1000],
            'land': [0, 0, 0],
            'wrong_fragment': [0, 0, 0],
            'urgent': [0, 0, 0],
            'hot': [0, 0, 0],
            'num_failed_logins': [0, 0, 0],
            'logged_in': [0, 0, 0],
            'num_compromised': [0, 0, 0],
            'root_shell': [0, 0, 0],
            'su_attempted': [0, 0, 0],
            'num_root': [0, 0, 0],
            'num_file_creations': [0, 0, 0],
            'num_shells': [0, 0, 0],
            'num_access_files': [0, 0, 0],
            'num_outbound_cmds': [0, 0, 0],
            'is_host_login': [0, 0, 0],
            'is_guest_login': [0, 0, 0],
            'count': [1, 2, 3],
            'srv_count': [1, 2, 3],
            'serror_rate': [0.0, 0.0, 0.0],
            'srv_serror_rate': [0.0, 0.0, 0.0],
            'rerror_rate': [0.0, 0.0, 0.0],
            'srv_rerror_rate': [0.0, 0.0, 0.0],
            'same_srv_rate': [1.0, 1.0, 1.0],
            'diff_srv_rate': [0.0, 0.0, 0.0],
            'srv_diff_host_rate': [0.0, 0.0, 0.0],
            'dst_host_count': [100, 100, 100],
            'dst_host_srv_count': [100, 100, 100],
            'dst_host_same_srv_rate': [1.0, 1.0, 1.0],
            'dst_host_diff_srv_rate': [0.0, 0.0, 0.0],
            'dst_host_same_src_port_rate': [0.0, 0.0, 0.0],
            'dst_host_srv_diff_host_rate': [0.0, 0.0, 0.0],
            'dst_host_serror_rate': [0.0, 0.0, 0.0],
            'dst_host_srv_serror_rate': [0.0, 0.0, 0.0],
            'dst_host_rerror_rate': [0.0, 0.0, 0.0],
            'dst_host_srv_rerror_rate': [0.0, 0.0, 0.0],
            'class': ['normal', 'anomaly', 'normal']
        })
        try:
            st.session_state.model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.le_class = train_model(sample_data)
            if st.session_state.model is None:
                st.error("Failed to initialize model. Using fallback mode.")
                logger.warning("Model initialization failed, proceeding without model")
            else:
                logger.info("Initialized model for ATC monitoring")
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            st.error(f"Model initialization error: {str(e)}")

    if not st.session_state.authenticated:
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NAMA IDPS Login</title>
            <style>
                :root {
                    --white: #e9e9e9;
                    --gray: #333;
                    --blue: #0367a6;
                    --lightblue: #008997;
                    --button-radius: 0.7rem;
                    --max-width: 758px;
                    --max-height: 420px;
                    font-size: 16px;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
                        Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
                }
                body {
                    align-items: center;
                    background-color: var(--white);
                    background: url("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/images/aeroplane.jpg");
                    background-attachment: fixed;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-size: cover;
                    display: grid;
                    height: 100vh;
                    place-items: center;
                    overflow: hidden;
                }
                .form__title {
                    font-weight: 300;
                    margin: 0;
                    margin-bottom: 1.25rem;
                }
                .link {
                    color: var(--gray);
                    font-size: 0.9rem;
                    margin: 1.5rem 0;
                    text-decoration: none;
                }
                .container {
                    background-color: var(--white);
                    border-radius: var(--button-radius);
                    box-shadow: 0 0.9rem 1.7rem rgba(0, 0, 0, 0.25),
                        0 0.7rem 0.7rem rgba(0, 0, 0, 0.22);
                    height: var(--max-height);
                    max-width: var(--max-width);
                    overflow: hidden;
                    position: relative;
                    width: 100%;
                }
                .container__form {
                    height: 100%;
                    position: absolute;
                    top: 0;
                    transition: all 0.6s ease-in-out;
                }
                .container--signin {
                    left: 0;
                    width: 50%;
                    z-index: 2;
                }
                .container.right-panel-active .container--signin {
                    transform: translateX(100%);
                }
                .container--signup {
                    left: 0;
                    opacity: 0;
                    width: 50%;
                    z-index: 1;
                }
                .container.right-panel-active .container--signup {
                    animation: show 0.6s;
                    opacity: 1;
                    transform: translateX(100%);
                    z-index: 5;
                }
                .container__overlay {
                    height: 100%;
                    left: 50%;
                    overflow: hidden;
                    position: absolute;
                    top: 0;
                    transition: transform 0.6s ease-in-out;
                    width: 50%;
                    z-index: 100;
                }
                .container.right-panel-active .container__overlay {
                    transform: translateX(-100%);
                }
                .overlay {
                    background: linear-gradient(to right, var(--lightblue), var(--blue));
                    background: url("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/images/aeroplane.jpg");
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
                }
                .container.right-panel-active .overlay {
                    transform: translateX(50%);
                }
                .overlay-panel {
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
                }
                .overlay--left {
                    transform: translateX(-20%);
                }
                .container.right-panel-active .overlay--left {
                    transform: translateX(0);
                }
                .overlay--right {
                    right: 0;
                    transform: translateX(0);
                }
                .container.right-panel-active .overlay--right {
                    transform: translateX(20%);
                }
                .btn {
                    background-color: var(--blue);
                    background-image: linear-gradient(90deg, var(--blue), var(--lightblue));
                    border-radius: 20px;
                    border: 1px solid var(--blue);
                    color: var(--white);
                    cursor: pointer;
                    font-size: 0.8rem;
                    font-weight: bold;
                    letter-spacing: 0.1rem;
                    padding: 0.9rem 4rem;
                    text-transform: uppercase;
                    transition: transform 80ms ease-in;
                }
                .form > .btn {
                    margin-top: 1.5rem;
                }
                .btn:active {
                    transform: scale(0.95);
                }
                .btn:focus {
                    outline: none;
                }
                .form {
                    background-color: var(--white);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    padding: 0 3rem;
                    height: 100%;
                    text-align: center;
                }
                .input {
                    background-color: #fff;
                    border: none;
                    padding: 0.9rem 0.9rem;
                    margin: 0.5rem 0;
                    width: 100%;
                }
                @keyframes show {
                    0%, 49.99% {
                        opacity: 0;
                        z-index: 1;
                    }
                    50%, 100% {
                        opacity: 1;
                        z-index: 5;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container" id="container">
                <!-- Sign Up -->
                <div class="container__form container--signup">
                    <form class="form" id="form1">
                        <h2 class="form__title">Sign Up</h2>
                        <input type="text" placeholder="Username" class="input" id="signupUsername" />
                        <input type="email" placeholder="Email" class="input" id="signupEmail" />
                        <input type="password" placeholder="Password" class="input" id="signupPassword" />
                        <button type="submit" class="btn">Sign Up</button>
                    </form>
                </div>
                <!-- Sign In -->
                <div class="container__form container--signin">
                    <form class="form" id="form2">
                        <h2 class="form__title">Sign In</h2>
                        <input type="email" placeholder="Email" class="input" id="signinEmail" />
                        <input type="password" placeholder="Password" class="input" id="signinPassword" />
                        <a href="#" class="link">Forgot your password?</a>
                        <button type="submit" class="btn">Sign In</button>
                    </form>
                </div>
                <!-- Overlay -->
                <div class="container__overlay">
                    <div class="overlay">
                        <div class="overlay-panel overlay--left">
                            <button class="btn" id="signInBtn">Sign In</button>
                        </div>
                        <div class="overlay-panel overlay--right">
                            <button class="btn" id="signUpBtn">Sign Up</button>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                const signInBtn = document.getElementById("signInBtn");
                const signUpBtn = document.getElementById("signUpBtn");
                const container = document.getElementById("container");
                const signInForm = document.getElementById("form2");
                const signUpForm = document.getElementById("form1");

                signUpBtn.addEventListener("click", () => {
                    container.classList.add("right-panel-active");
                });

                signInBtn.addEventListener("click", () => {
                    container.classList.remove("right-panel-active");
                });

                signUpForm.addEventListener("submit", (e) => {
                    e.preventDefault();
                    const username = document.getElementById("signupUsername").value;
                    const email = document.getElementById("signupEmail").value;
                    const password = document.getElementById("signupPassword").value;
                    fetch("/_stcore/streamlit_script_run", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            "signup": {
                                "username": username,
                                "email": email,
                                "password": password
                            }
                        })
                    }).then(response => {
                        if (response.ok) {
                            alert("Sign-up successful! Please sign in.");
                            container.classList.remove("right-panel-active");
                        } else {
                            alert("Sign-up failed. Username may already exist.");
                        }
                    }).catch(error => {
                        alert("Error during sign-up: " + error.message);
                    });
                });

                signInForm.addEventListener("submit", (e) => {
                    e.preventDefault();
                    const email = document.getElementById("signinEmail").value;
                    const password = document.getElementById("signinPassword").value;
                    fetch("/_stcore/streamlit_script_run", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            "signin": {
                                "email": email,
                                "password": password
                            }
                        })
                    }).then(response => {
                        if (response.ok) {
                            window.location.reload();
                        } else {
                            alert("Sign-in failed. Please check your credentials.");
                        }
                    }).catch(error => {
                        alert("Error during sign-in: " + error.message);
                    });
                });
            </script>
        </body>
        </html>
        """
        components.html(html_content, height=500)

        # Handle form submissions
        try:
            if "_streamlit_script_run_data" in st.session_state:
                script_run_data = st.session_state.get("_streamlit_script_run_data", {})
                if "signup" in script_run_data:
                    signup_data = script_run_data["signup"]
                    username = signup_data["username"]
                    password = signup_data["password"]
                    if register_user(username, password):
                        st.success("Registration successful! Please sign in.")
                        log_user_activity(username, "Registered")
                    else:
                        st.error("Registration failed. Username may already exist.")
                elif "signin" in script_run_data:
                    signin_data = script_run_data["signin"]
                    username = signin_data["username"]
                    password = signin_data["password"]
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        log_user_activity(username, "Signed in")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password")
        except Exception as e:
            logger.error(f"Error processing form submission: {str(e)}")
            st.error(f"Error processing form submission: {str(e)}")
    else:
        # Main application logic after authentication
        st.sidebar.image("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/images/logo.png", use_column_width=True, caption="NAMA IDPS")
        st.sidebar.markdown("<h2 style='text-align: center; color: #E6E6FA;'>Navigation</h2>", unsafe_allow_html=True)
        
        page = st.sidebar.radio("", [
            "🏠 Dashboard",
            "✈️ ATC Monitoring",
            "🛸 Drone Surveillance",
            "🛡️ Threat Intelligence",
            "🔍 Compliance Monitoring",
            "📊 Reports",
            "⚙️ Settings"
        ])

        if 'adsb_running' not in st.session_state:
            st.session_state.adsb_running = True
            threading.Thread(target=periodic_adsb_fetch, args=(10,), daemon=True).start()
        if 'radar_running' not in st.session_state:
            st.session_state.radar_running = True
            threading.Thread(target=periodic_radar_update, args=(5,), daemon=True).start()
        if 'nmap_running' not in st.session_state:
            st.session_state.nmap_running = True
            threading.Thread(target=periodic_nmap_scan, args=("192.168.1.0/24", "TCP SYN", "1-1000"), daemon=True).start()
        if 'drone_running' not in st.session_state:
            st.session_state.drone_running = True
            threading.Thread(target=periodic_drone_detection, daemon=True).start()
        if 'threat_running' not in st.session_state:
            st.session_state.threat_running = True
            threading.Thread(target=periodic_threat_fetch, args=(None,), daemon=True).start()
        if 'compliance_running' not in st.session_state:
            st.session_state.compliance_running = True
            threading.Thread(target=periodic_compliance_check, daemon=True).start()

        if page == "🏠 Dashboard":
            st.markdown("<h1 style='text-align: center;'>NAMA Intrusion Detection & Prevention System</h1>", unsafe_allow_html=True)
            st.markdown("<div class='card'><h3>System Overview</h3></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Recent Alerts")
                if st.session_state.alert_log:
                    st.dataframe(pd.DataFrame(st.session_state.alert_log[-5:]))
                else:
                    st.info("No alerts to display.")
            with col2:
                st.markdown("#### System Status")
                st.metric("Active Threats", len(st.session_state.threats))
                st.metric("Flight Conflicts", len(st.session_state.flight_conflicts))
                st.metric("Unauthorized Drones", sum(d['status'] == 'unidentified' for d in st.session_state.drone_results))

        elif page == "✈️ ATC Monitoring":
            st.markdown("<h1>ATC Monitoring</h1>", unsafe_allow_html=True)
            if st.session_state.radar_data:
                fig = display_radar(st.session_state.radar_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            if st.session_state.atc_results:
                st.markdown("#### Recent ATC Data")
                st.dataframe(pd.DataFrame(st.session_state.atc_results))
            if not st.session_state.atc_anomalies.empty:
                st.markdown("#### Detected Anomalies")
                st.dataframe(st.session_state.atc_anomalies)

        elif page == "🛸 Drone Surveillance":
            st.markdown("<h1>Drone Surveillance</h1>", unsafe_allow_html=True)
            display_drone_data()

        elif page == "🛡️ Threat Intelligence":
            st.markdown("<h1>Threat Intelligence</h1>", unsafe_allow_html=True)
            display_threat_intelligence()

        elif page == "🔍 Compliance Monitoring":
            st.markdown("<h1>Compliance Monitoring</h1>", unsafe_allow_html=True)
            display_compliance_metrics()

        elif page == "📊 Reports":
            st.markdown("<h1>Reports</h1>", unsafe_allow_html=True)
            st.markdown("<div class='card'>")
            st.markdown("### Generate Report")
            if st.button("Generate PDF Report", key="generate_report"):
                download_report()
            st.markdown("</div>", unsafe_allow_html=True)

        elif page == "⚙️ Settings":
            st.markdown("<h1>Settings</h1>", unsafe_allow_html=True)
            st.markdown("<div class='card'>")
            st.markdown("### System Configuration")
            target = st.text_input("NMAP Scan Target", "192.168.1.0/24")
            scan_type = st.selectbox("Scan Type", ["TCP SYN", "TCP Connect", "UDP"])
            port_range = st.text_input("Port Range", "1-1000")
            custom_args = st.text_input("Custom NMAP Arguments", "")
            if st.button("Run NMAP Scan"):
                results = run_nmap_scan(target, scan_type, port_range, custom_args)
                st.session_state.scan_results = results
                st.dataframe(pd.DataFrame(results))
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
