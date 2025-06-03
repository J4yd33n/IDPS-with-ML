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
                    symbol='cross',
                    line=dict(width=2, color=WICKET_THEME['text']),
                    opacity=0.5
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
        nm = nmap.PortScanner()
        scan_args = {'TCP SYN': '-sS', 'TCP Connect': '-sT', 'UDP': '-sU'}
        args = f"{scan_args[scan_type]} {custom_args}"
        
        scan_results = []
        
        nm.scan(target, port_range, arguments=args)
        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                for port in nm[host][proto].keys():
                    state = nm[host][proto][port]['state']
                    service = nm[host][proto][port].get('name', 'unknown')
                    scan_results.append({
                        'port': port,
                        'protocol': proto,
                        'state': state,
                        'service': service
                    })
        
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
                'protocol': proto,
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
                        'dst_host_srt_diff_host_rate': np.random.randint(0, 100),
                        'dst_host_serror_rate': 0.0,
                        'dst_host_srt_serror_rate': np.random.randint(0, 255),
                        'dst_host_rerror_rate': 0.0,
                        'dst_host_srt_rerror_rate': np.random.randint(0, 255),
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
            'icao24': [f"ICAO{i:04d}" for i in range(num_samples)],
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
            'same_srv_rate': np.random.uniform(0, 10, num_samples),
            'diff_sv_rate': np.random.diff(0, num_samples, num_samples),
            'srv_diff_host_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_count': np.random.randint(1, 10, num_samples),
            'dst_host_srt_count': np.random.randint(1, 100, num_samples),
            'dst_host_same_srt_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_diff_srt_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_same_src_port_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srt_diff_host_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_serror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srt_serror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_rerror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srt_rerror_rate': np.random.uniform(0, 1, num_samples),
            'latitude': np.random.uniform(region['lat_min'], region['lat_max'], num_samples),
            'longitude': np.random.uniform(region['lon_min'], region['lon_max'], num_samples),
            'altitude': np.random.uniform(0, 40000, num_samples),
            'velocity': np.random.uniform(0, 600, num_samples)
        }
        log_user_activity("system", "Simulated ATC traffic")
        return pd.DataFrame(data).to_dict('records')

# Anomaly Detection in Air Traffic Data
def detect_air_traffic_anomalies(df):
    try:
        features = ['latitude', 'longitude', 'altitude', 'velocity']
        X = df[features].fillna(0)
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        predictions = model.predict(X)
        df['anomaly'] = predictions == -1
        anomalies = df[df['anomaly'] == True]
        return df, anomalies
    except Exception as e:
        logger.error(f"Flight data processing error: {str(e)}")
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
                vel2 = row2['velocity'] * np.array([np.cos(np.radians(np.radians(row2.get('heading', 0)))), np.sin(np.radians(np.radians(row2.get('heading', np0))))])
                rel_vel = np.linalg.norm(vel1 - vel2)
                time_to_collision = dist / (rel_vel / 3600) if rel_vel > 0 else float('inf')
                if dist < distance_threshold and time_to_collision < time_threshold:
                    risks.append({
                        'icao24_1': row1['icao24'],
                        'icao24_2': row2['icao24'],
                        'distance_km': dist,
                        'time_to_collision': time_to_collision,
                        'severity': 'critical' if time_to_collision < 60 else 'high'
                    })
        if risks:
            for risk in risks:
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Flight Conflict',
                    'severity': risk['severity'],
                    'details': f'Potential collision between {risk['icao24_1']} and {risk['icao24_2']}: {risk['distance_km']:.2f}km, {risk['time_to_collision']:.0f}s'
                })
        log_user_activity("system", f"Detected {len(risks)} flight conflicts")
        return risks
    except Exception as e:
        logger.error(f"Collision risk detection error: {str(e)}")
        return st.error(f"Collision risk detection error: {str(e)}")
        return []

# Flight Data Processing: Traffic Flow Optimization
def optimize_traffic_flow(traffic_data, num_clusters=3):
    try:
        df = pd.DataFrame(traffic_data)
        if len(df) < 2:
            return []        )
        features = ['latitude', 'longitude', 'altitude']
        X = df[features].values().fillna(0)
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
                'icao24_list': cluster_df['icao24'].to_list(),
                'suggested_route': centroid,
                'aircraft_count': len(cluster_df)
            })
        log_user_activity("system", f"Optimized routes for {len(routes)} clusters")
        return routes
    except Exception as e:
        logger.error(f"Traffic optimization error: {str(e)}")
        return str(e.error(f"Traffic optimization error: {str(e)}"))

# Drone Detection
def periodic_drone_detection(interval=120):
    try:
        while 'drone_running' in st.session_state and st.session_state.drone_running:
            region = {'lat_min': 4, 'lat_max': 14, 'lon_min': 2, 'lon_max': 15}
            num_drones = np.random.randint(0, 5)
            drone_results = []
            for i in range(num_drones):
                altitude = np.random.uniform(0, 1000)
                is_unauthorized = altitude < 400 and np.random.random() > 0.7
                drone_results.append({
                    'timestamp': pd.Timestamp.now(),
                    'drone_id': f"DRN{i:03d}",
                    'latitude': np.random.uniform(region['lat_min'], region['lat_max']),
                    'longitude': np.random.uniform(region['lon_min'], region['lon_max']),
                    'altitude': altitude,
                    'status': 'ununauthorized' if is_unauthorized else 'authorized',
                    'severity': 'high' if is_unauthorized else 'low'
                })
            st.session_state.drone_results = drone_results
            if any(d['status'] == 'ununauthorized' for d in drone_results):
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Drone Intrusion',
                    'severity': 'high',
                    'details': f"Detected {sum(d['status'] == 'ununauthorized' for d in drone_results)} unauthorized drones"
                })
            log_user_activity("system", f"Drone detection: {len(drone_results)} drones, {sum(d['status'] == 'unapproved' for d in drone_results)} unauthorized")
            time.sleep(interval)
    except Exception as e:
        logger.error(f"Drone detection error: {str(e)}")
        st.session_state.alert_log.append({
            'timestamp': datetime.now(),
            'type': 'Drone Detection Error',
            'severity': 'high',
            'details': f"Drone detection failed: {str(e)}"
        })

# Threat Intelligence Integration
@st.cache_data(ttl=3600)
def fetch_threat_feed(api_key=None, source="alienvault"):
    """
    Fetch threat intelligence from external sources like AlienVault OTX or a simulated feed.
    """
    try:
        if source == "alienvault" and api_key:
            url = "https://otx.alienvault.com/api/v1/pulses/subscribed"
            headers = {"X-OTX-API-KEY": api_key}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get("results", [])
                threats = []
                for pulse in data[:10]:  # Limit to 10 for performance
                    threats.append({
                        'timestamp': pd.Timestamp.now(),
                        'threat_id': pulse.get('id', 'unknown'),
                        'description': pulse.get('name', 'No description'),
                        'indicators': [i.get('indicator') for i in pulse.get('indicators', [])],
                        'severity': 'medium',
                        'source': 'alienvault'
                    })
                log_user_activity("system", f"Fetched {len(threats)} threats from AlienVault")
                return threats
            else:
                logger.warning(f"AlienVault API error: {response.status_code}")
                st.warning("Failed to fetch AlienVault threat feed. Using simulated data.")
        return simulate_threat_feed()
    except Exception as e:
        logger.error(f"Threat feed fetch error: {str(e)}")
        st.warning(f"Threat feed error: {str(e)}. Using simulated data.")
        return simulate_threat_feed()

def simulate_threat_feed(num_threats=5):
    """
    Simulate a threat intelligence feed for testing or fallback.
    """
    try:
        threats = []
        for i in range(num_threats):
            threats.append({
                'timestamp': pd.Timestamp.now(),
                'threat_id': f"THR{i:03d}",
                'description': f"Simulated threat {i+1}",
                'indicators': [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"],
                'severity': np.random.choice(['low', 'medium', 'high']),
                'source': 'simulated'
            })
        log_user_activity("system", f"Simulated {num_threats} threats")
        return threats
    except Exception as e:
        logger.error(f"Threat simulation error: {str(e)}")
        return []

def periodic_threat_fetch(api_key, interval=3600):
    """
    Periodically fetch threat intelligence and update session state.
    """
    try:
        while 'threat_running' in st.session_state and st.session_state.threat_running:
            threats = fetch_threat_feed(api_key)
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
            'details': f"Threat feed update failed: {str(e)}"
        })

# Model Training and Prediction
def train_model(df, model_type="xgboost"):
    """
    Train a machine learning model for traffic classification.
    """
    try:
        label_encoders = {}
        le_class = LabelEncoder()
        df_processed, label_encoders, le_class = preprocess_data(df, label_encoders, le_class, is_train=True)
        if df_processed is None:
            return None, None, None, None

        X = df_processed.drop(['class'], axis=1, errors='ignore')
        y = df_processed['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == "xgboost":
            model = XGBClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)
        elif model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train_scaled, y_train)
        elif model_type == "lstm":
            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            model = Sequential([
                LSTM(50, input_shape=(1, X_train_scaled.shape[1]), return_sequences=False),
                Dense(len(np.unique(y)), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        accuracy = model.score(X_test_scaled, y_test) if model_type != "lstm" else model.evaluate(
            X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1])), y_test, verbose=0)[1]
        logger.info(f"Trained {model_type} model with accuracy: {accuracy:.2f}")
        return model, scaler, label_encoders, le_class
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        st.error(f"Model training error: {str(e)}")
        return None, None, None, None

def predict_traffic(df, model, scaler, label_encoders, le_class):
    """
    Predict traffic classification using the trained model.
    """
    try:
        df_processed, _, _ = preprocess_data(df, label_encoders, le_class, is_train=False)
        if df_processed is None:
            return "unknown", 0.0

        X = df_processed.drop(['class'], axis=1, errors='ignore')
        X_scaled = scaler.transform(X)

        if isinstance(model, Sequential):  # LSTM model
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            probabilities = model.predict(X_reshaped, verbose=0)
            prediction = le_class.inverse_transform([np.argmax(probabilities[0])])[0]
            confidence = np.max(probabilities[0])
        else:  # XGBoost or RandomForest
            probabilities = model.predict_proba(X_scaled)
            prediction = le_class.inverse_transform([np.argmax(probabilities[0])])[0]
            confidence = np.max(probabilities[0])

        if confidence < 0.5:
            st.session_state.alert_log.append({
                'timestamp': datetime.now(),
                'type': 'Low Confidence Prediction',
                'severity': 'medium',
                'details': f"Prediction {prediction} with low confidence {confidence:.2f}"
            })
        return prediction, confidence
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "error", 0.0

# Report Generation
def generate_pdf_report(data, filename="nama_idps_report.pdf"):
    """
    Generate a PDF report summarizing IDPS activities.
    """
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("NAMA IDPS Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Summary
        elements.append(Paragraph("Summary", styles['Heading2']))
        summary_data = [
            ["Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Total Alerts", len(st.session_state.alert_log)],
            ["High Severity Alerts", sum(1 for a in st.session_state.alert_log if a['severity'] == 'high')],
            ["Flight Conflicts", len(st.session_state.flight_conflicts)],
            ["Unauthorized Drones", sum(1 for d in st.session_state.drone_results if d['status'] == 'unauthorized')]
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

        # Alerts
        elements.append(Paragraph("Recent Alerts", styles['Heading2']))
        alert_data = [["Timestamp", "Type", "Severity", "Details"]]
        for alert in st.session_state.alert_log[-5:]:  # Last 5 alerts
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
        log_user_activity("system", f"Generated PDF report: {filename}")
        return filename
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        st.error(f"PDF generation error: {str(e)}")
        return None

# Main Streamlit App
def main():
    """
    Main Streamlit application for NAMA IDPS.
    """
    apply_wicket_css()
    setup_user_db()

    if not st.session_state.authenticated:
        st.markdown("""
            <div class="auth-container">
                <div class="auth-overlay"></div>
                <div class="auth-card">
                    <div class="auth-form">
                        <h2 class="auth-neon-text">NAMA IDPS Login</h2>
                    </div>
                </div>
                <div class="radar"></div>
            </div>
        """, unsafe_allow_html=True)

        with st.container():
            username = st.text_input("Username", placeholder="Enter username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", key="login_button"):
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        log_user_activity(username, "Logged in")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password")
            with col2:
                if st.button("Register", key="register_button"):
                    st.session_state.show_register = True

            if 'show_register' in st.session_state and st.session_state.show_register:
                st.markdown("<h3 class='auth-neon-text'>Register New User</h3>", unsafe_allow_html=True)
                new_username = st.text_input("New Username", placeholder="Enter new username", key="reg_username")
                new_password = st.text_input("New Password", type="password", placeholder="Enter new password", key="reg_password")
                if st.button("Create Account", key="create_account"):
                    if register_user(new_username, new_password):
                        st.success("User registered successfully! Please login.")
                        st.session_state.show_register = False
                    else:
                        st.error("Username already exists or registration failed.")

    else:
        st.sidebar.markdown(f"""
            <div style='text-align: center;'>
                <img src='https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/logo.png' class='logo-image'>
                <h2 style='color: {WICKET_THEME["text_light"]};'>NAMA IDPS</h2>
            </div>
        """, unsafe_allow_html=True)

        menu = ["Dashboard", "Network Scan", "ATC Monitoring", "Drone Detection", "Threat Intelligence", "Reports", "Logout"]
        choice = st.sidebar.selectbox("Menu", menu, format_func=lambda x: f"<i class='fas fa-{x.lower().replace(' ', '-')}'> {x}</i>", key="sidebar_menu")

        if choice == "Dashboard":
            st.markdown("<h1>System Dashboard</h1>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3>Compliance Metrics</h3>", unsafe_allow_html=True)
                st.json(st.session_state.compliance_metrics)
            with col2:
                st.markdown("<h3>Recent Alerts</h3>", unsafe_allow_html=True)
                if st.session_state.alert_log:
                    alert_df = pd.DataFrame(st.session_state.alert_log[-5:])
                    st.dataframe(alert_df[["timestamp", "type", "severity", "details"]])
                else:
                    st.info("No alerts yet.")

        elif choice == "Network Scan":
            st.markdown("<h1>Network Scan</h1>", unsafe_allow_html=True)
            target = st.text_input("Target IP/Host", "192.168.1.1")
            scan_type = st.selectbox("Scan Type", ["TCP SYN", "TCP Connect", "UDP"])
            port_range = st.text_input("Port Range", "1-1024")
            custom_args = st.text_input("Custom NMAP Args", "-T4")
            if st.button("Start Scan"):
                st.session_state.nmap_running = True
                threading.Thread(target=periodic_nmap_scan, args=(target, scan_type, port_range, custom_args), daemon=True).start()
            if st.session_state.scan_results:
                st.dataframe(pd.DataFrame(st.session_state.scan_results))

        elif choice == "ATC Monitoring":
            st.markdown("<h1>ATC Monitoring</h1>", unsafe_allow_html=True)
            num_samples = st.slider("Number of Samples", 5, 50, 10)
            if st.button("Start ATC Monitoring"):
                st.session_state.adsb_running = True
                st.session_state.radar_running = True
                threading.Thread(target=periodic_adsb_fetch, args=(num_samples, None), daemon=True).start()
                threading.Thread(target=periodic_radar_update, args=(5, None), daemon=True).start()
            if st.session_state.radar_data:
                fig = display_radar(st.session_state.radar_data)
                if fig:
                    st.plotly_chart(fig)
            if st.session_state.flight_conflicts:
                st.warning(f"Detected {len(st.session_state.flight_conflicts)} potential collisions!")
                st.dataframe(pd.DataFrame(st.session_state.flight_conflicts))

        elif choice == "Drone Detection":
            st.markdown("<h1>Drone Detection</h1>", unsafe_allow_html=True)
            if st.button("Start Drone Monitoring"):
                st.session_state.drone_running = True
                threading.Thread(target=periodic_drone_detection, daemon=True).start()
            if st.session_state.drone_results:
                st.dataframe(pd.DataFrame(st.session_state.drone_results))

        elif choice == "Threat Intelligence":
            st.markdown("<h1>Threat Intelligence</h1>", unsafe_allow_html=True)
            api_key = st.text_input("API Key (AlienVault OTX)", type="password")
            if st.button("Fetch Threats"):
                st.session_state.threat_running = True
                threading.Thread(target=periodic_threat_fetch, args=(api_key,), daemon=True).start()
            if st.session_state.threats:
                st.dataframe(pd.DataFrame(st.session_state.threats))

        elif choice == "Reports":
            st.markdown("<h1>Generate Reports</h1>", unsafe_allow_html=True)
            if st.button("Generate PDF Report"):
                filename = generate_pdf_report(None)
                if filename:
                    with open(filename, "rb") as f:
                        st.download_button("Download Report", f, file_name=filename)

        elif choice == "Logout":
            st.session_state.authenticated = False
            st.session_state.username = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
