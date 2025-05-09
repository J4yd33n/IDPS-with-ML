import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import logging
import os
import sqlite3
import base64
import io
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from transformers import pipeline
import re
from collections import deque

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

# Wicket-inspired theme with new color palette
WICKET_THEME = {
    "primary_bg": "#1A1F36",  # Midnight Blue
    "secondary_bg": "#2D3748",
    "accent": "#3B82F6",      # Electric Blue
    "text": "#F7FAFC",
    "text_light": "#FFFFFF",
    "card_bg": "rgba(255, 255, 255, 0.05)",
    "border": "#4A5568",
    "button_bg": "#3B82F6",
    "button_text": "#FFFFFF",
    "hover": "#2563EB",
    "error": "#EF4444",       # Soft Red
    "success": "#10B981"      # Emerald Green
}

# Apply updated CSS with Poppins and Inter fonts
def apply_wicket_css(theme_mode='dark'):
    css = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700&family=Inter:wght@400;500&display=swap');
            .stApp {{
                background-color: {WICKET_THEME['primary_bg']};
                color: {WICKET_THEME['text']};
                font-family: 'Inter', sans-serif;
            }}
            .css-1d391kg {{
                background-color: {WICKET_THEME['secondary_bg']};
                color: {WICKET_THEME['text']};
                padding: 20px;
                border-right: 1px solid {WICKET_THEME['border']};
            }}
            .css-1d391kg .stSelectbox, .css-1d391kg .stButton>button {{
                background-color: {WICKET_THEME['card_bg']};
                color: {WICKET_THEME['text']};
                border-radius: 8px;
                border: 1px solid {WICKET_THEME['border']};
                font-family: 'Inter', sans-serif;
            }}
            .css-1d391kg .stButton>button {{
                background-color: {WICKET_THEME['button_bg']};
                color: {WICKET_THEME['button_text']};
                transition: background-color 0.3s;
            }}
            .css-1d391kg .stButton>button:hover {{
                background-color: {WICKET_THEME['hover']};
            }}
            .main .block-container {{
                padding: 30px;
                max-width: 1200px;
                margin: auto;
            }}
            .card {{
                background: {WICKET_THEME['card_bg']};
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 5px 5px 15px rgba(0,0,0,0.4), -5px -5px 15px rgba(255,255,255,0.1);
                transition: transform 0.2s;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .stTextInput>div>input, .stSelectbox>div>select {{
                background-color: {WICKET_THEME['card_bg']};
                border: 1px solid {WICKET_THEME['border']};
                border-radius: 8px;
                padding: 10px;
                color: {WICKET_THEME['text']};
                font-family: 'Inter', sans-serif;
            }}
            .stTextInput input:focus {{
                border-color: {WICKET_THEME['accent']};
                box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
            }}
            .stButton>button {{
                background-color: {WICKET_THEME['button_bg']};
                color: {WICKET_THEME['button_text']};
                border-radius: 8px;
                padding: 10px 20px;
                border: none;
                transition: background-color 0.3s;
                font-family: 'Poppins', sans-serif;
            }}
            .stButton>button:hover {{
                background-color: {WICKET_THEME['hover']};
            }}
            .plotly-graph-div {{
                background-color: {WICKET_THEME['card_bg']};
                border-radius: 8px;
                padding: 10px;
            }}
            .logo-image {{
                width: 100%;
                max-width: 120px;
                height: auto;
                margin-bottom: 20px;
            }}
            h1, h2, h3 {{
                font-family: 'Poppins', sans-serif;
                color: {WICKET_THEME['text']};
                font-weight: 500;
            }}
            .stAlert {{
                border-radius: 8px;
                padding: 15px;
                color: {WICKET_THEME['text']};
                background-color: {WICKET_THEME['error']};
            }}
            p, li, div, span {{
                color: {WICKET_THEME['text']};
                font-family: 'Inter', sans-serif;
            }}
            @keyframes shake {{
                0%, 100% {{ transform: translateX(0); }}
                20%, 60% {{ transform: translateX(-10px); }}
                40%, 80% {{ transform: translateX(10px); }}
            }}
            .login-card {{
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 2rem;
                max-width: 400px;
                margin: 0 auto;
            }}
            .shake {{
                animation: shake 0.4s ease-in-out;
            }}
            .particles-bg {{
                position: fixed;
                inset: 0;
                background: linear-gradient(135deg, #1A1F36 0%, #2D3748 100%);
                background-image: radial-gradient(circle, rgba(59, 130, 246, 0.2) 2px, transparent 2px);
                background-size: 20px 20px;
                z-index: -1;
            }}
            .glitch {{
                position: relative;
                color: #F7FAFC;
            }}
            .glitch:before, .glitch:after {{
                content: 'NAMA IDPS';
                position: absolute;
                top: 0;
                left: 0;
                opacity: 0.8;
            }}
            .glitch:before {{
                color: #3B82F6;
                animation: glitch 1s infinite;
                transform: translate(-2px, 2px);
            }}
            .glitch:after {{
                color: #EF4444;
                animation: glitch 1.5s infinite;
                transform: translate(2px, -2px);
            }}
            @keyframes glitch {{
                0% {{ opacity: 0.8; }}
                50% {{ opacity: 0.4; }}
                100% {{ opacity: 0.8; }}
            }}
            /* Adapted CSS from provided code */
            .auth-container {{
                background-color: rgba(233, 233, 233, 0.9);
                border-radius: 0.7rem;
                box-shadow: 0 0.9rem 1.7rem rgba(0, 0, 0, 0.25),
                    0 0.7rem 0.7rem rgba(0, 0, 0, 0.22);
                max-width: 758px;
                height: 420px;
                overflow: hidden;
                position: relative;
                width: 100%;
                margin: 0 auto;
            }}
            .auth-form {{
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                padding: 0 3rem;
                height: 100%;
                text-align: center;
            }}
            .auth-form h2 {{
                font-weight: 300;
                margin: 0;
                margin-bottom: 1.25rem;
                color: #333;
            }}
            .auth-input {{
                background-color: #fff;
                border: none;
                padding: 0.9rem 0.9rem;
                margin: 0.5rem 0;
                width: 100%;
                border-radius: 0.3rem;
            }}
            .auth-btn {{
                background-image: linear-gradient(90deg, #0367a6 0%, #008997 74%);
                border-radius: 20px;
                border: 1px solid #0367a6;
                color: #e9e9e9;
                cursor: pointer;
                font-size: 0.8rem;
                font-weight: bold;
                letter-spacing: 0.1rem;
                padding: 0.9rem 4rem;
                text-transform: uppercase;
                transition: transform 80ms ease-in;
                margin-top: 1.5rem;
            }}
            .auth-btn:hover {{
                background-image: linear-gradient(90deg, #024e7a 0%, #006b76 74%);
            }}
            .auth-btn:active {{
                transform: scale(0.95);
            }}
            .auth-link {{
                color: #333;
                font-size: 0.9rem;
                margin: 1.5rem 0;
                text-decoration: none;
            }}
            .auth-link:hover {{
                color: #0367a6;
            }}
        </style>
        <div class="particles-bg"></div>
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
    # Check if default user exists, if not, create it
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
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hashed))
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

# Real-time NMAP scan
def run_nmap_scan(target, scan_type, port_range, custom_args, callback):
    try:
        if not NMAP_AVAILABLE:
            raise ImportError("python-nmap library is not installed.")
        nm = nmap.PortScannerAsync()
        scan_args = {'TCP SYN': '-sS', 'TCP Connect': '-sT', 'UDP': '-sU'}
        args = f"{scan_args[scan_type]} {custom_args}"
        nm.scan(target, port_range, arguments=args, callback=callback)
        while nm.still_scanning():
            st.spinner("Scanning in progress...")
        results = []
        for host in nm.all_hosts():
            for proto in nm[host].all_protocols():
                for port in nm[host][proto].keys():
                    state = nm[host][proto][port]['state']
                    service = nm[host][proto][port].get('name', 'unknown')
                    results.append({
                        'port': port,
                        'protocol': proto,
                        'state': state,
                        'service': service
                    })
        log_user_activity("system", f"Real-time NMAP scan on {target}")
        return results
    except Exception as e:
        logger.error(f"NMAP scan error: {str(e)}")
        st.error(f"NMAP scan error: {str(e)}")
        return []

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

# Fetch real ADS-B data
def fetch_adsb_data(num_samples=10):
    try:
        username = st.secrets.get('OPENSKY_USERNAME', os.getenv('OPENSKY_USERNAME'))
        password = st.secrets.get('OPENSKY_PASSWORD', os.getenv('OPENSKY_PASSWORD'))
        
        if not username or not password:
            raise ValueError("OpenSky credentials not provided.")
        
        url = "https://opensky-network.org/api/states/all"
        response = requests.get(url, auth=(username, password))
        if response.status_code != 200:
            raise Exception(f"Failed to fetch ADS-B data: {response.status_code}")
        data = response.json()['states'][:num_samples]
        airports = ['DNMM', 'DNAA', 'DNKN', 'DNPO']
        adsb_records = []
        for state in data:
            adsb_records.append({
                'timestamp': pd.Timestamp.now(),
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
                'latitude': state[6] if state[6] is not None else 0.0,
                'longitude': state[5] if state[5] is not None else 0.0,
                'altitude': state[7] if state[7] is not None else 0.0,
                'velocity': state[9] if state[9] is not None else 0.0
            })
        log_user_activity("system", "Fetched real ADS-B data")
        return adsb_records
    except Exception as e:
        logger.error(f"ADS-B fetch error: {str(e)}")
        st.error(f"ADS-B fetch error: {str(e)}")
        return []

# Simulate aviation traffic
def simulate_aviation_traffic(num_samples=10):
    try:
        airports = ['DNMM', 'DNAA', 'DNHT', 'DNPO']
        data = {
            'timestamp': pd.date_range(start='now', periods=num_samples, freq='S'),
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
            'latitude': np.random.uniform(4, 14, num_samples),
            'longitude': np.random.uniform(2, 15, num_samples),
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

# RF Signal Intrusion Detection (Simulated)
def detect_rf_signal_intrusion(signal_data):
    try:
        model = RandomForestClassifier(random_state=42)
        X = signal_data[['frequency', 'amplitude', 'noise_level']]
        y = signal_data['label']
        model.fit(X, y)
        predictions = model.predict(X)
        signal_data['intrusion'] = predictions
        intrusions = signal_data[signal_data['intrusion'] == 1]
        log_user_activity("system", f"Detected {len(intrusions)} RF signal intrusions")
        return signal_data, intrusions
    except Exception as e:
        logger.error(f"RF signal intrusion detection error: {str(e)}")
        st.error(f"RF signal intrusion detection error: {str(e)}")
        return signal_data, pd.DataFrame()

# Insider Threat Detection
def detect_insider_threats(username, actions, sequence_length=5):
    try:
        if username not in st.session_state.user_activity:
            st.session_state.user_activity[username] = deque(maxlen=sequence_length)
        st.session_state.user_activity[username].append(actions)
        
        if len(st.session_state.user_activity[username]) < sequence_length:
            return False, 0.0
        
        model = Sequential([
            LSTM(50, input_shape=(sequence_length, 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        X = np.array(list(st.session_state.user_activity[username])).reshape(1, sequence_length, 1)
        score = model.predict(X)[0][0]
        is_threat = score > 0.7
        log_user_activity(username, f"Insider threat score: {score:.2f}")
        return is_threat, score
    except Exception as e:
        logger.error(f"Insider threat detection error: {str(e)}")
        st.error(f"Insider threat detection error: {str(e)}")
        return False, 0.0

# Predictive Maintenance with Security Alerts
def predict_maintenance_anomalies(equipment_data):
    try:
        features = ['temperature', 'vibration', 'uptime']
        X = equipment_data[features].fillna(0)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        predictions = model.predict(X)
        equipment_data['anomaly'] = predictions == -1
        anomalies = equipment_data[equipment_data['anomaly']]
        log_user_activity("system", f"Detected {len(anomalies)} equipment anomalies")
        return equipment_data, anomalies
    except Exception as e:
        logger.error(f"Equipment anomaly detection error: {str(e)}")
        st.error(f"Equipment anomaly detection error: {str(e)}")
        return equipment_data, pd.DataFrame()

# Real-time Cyber Threat Intelligence
def analyze_threat_feeds(text_data):
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = ["threat", "benign"]
        results = classifier(text_data, labels, multi_label=False)
        threats = [text for text, score in zip(text_data, results['scores']) if score > 0.7 and results['labels'][score.index()] == "threat"]
        log_user_activity("system", f"Detected {len(threats)} cyber threats")
        return threats
    except Exception as e:
        logger.error(f"Threat intelligence error: {str(e)}")
        st.error(f"Threat intelligence error: {str(e)}")
        return []

# Autonomous Drone Intrusion Classification
def classify_drone_intrusion(signal_data):
    try:
        model = RandomForestClassifier(random_state=42)
        X = signal_data[['frequency', 'power_level']]
        y = signal_data['label']
        model.fit(X, y)
        predictions = model.predict(X)
        signal_data['intrusion'] = predictions
        intrusions = signal_data[signal_data['intrusion'] == 1]
        log_user_activity("system", f"Detected {len(intrusions)} drone intrusions")
        return signal_data, intrusions
    except Exception as e:
        logger.error(f"Drone intrusion classification error: {str(e)}")
        st.error(f"Drone intrusion classification error: {str(e)}")
        return signal_data, pd.DataFrame()

# ML-based Log File Analyzer for SCADA
def analyze_scada_logs(logs, sequence_length=10):
    try:
        model = Sequential([
            LSTM(100, input_shape=(sequence_length, 1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        sequences = []
        for i in range(len(logs) - sequence_length):
            sequences.append(logs[i:i+sequence_length])
        X = np.array(sequences).reshape(-1, sequence_length, 1)
        predictions = model.predict(X)
        anomalies = [logs[i+sequence_length] for i, p in enumerate(predictions) if p > 0.7]
        log_user_activity("system", f"Detected {len(anomalies)} SCADA log anomalies")
        return anomalies
    except Exception as e:
        logger.error(f"SCADA log analysis error: {str(e)}")
        st.error(f"SCADA log analysis error: {str(e)}")
        return []

# Cyber-Physical System Attack Simulation & Detection
def simulate_cps_attack(data, attack_type='mitm'):
    try:
        if attack_type == 'mitm':
            data['latency'] = data['latency'] * np.random.uniform(1.5, 3.0, len(data))
        model = IsolationForest(contamination=0.1, random_state=42)
        X = data[['latency', 'packet_loss']]
        model.fit(X)
        predictions = model.predict(X)
        data['attack'] = predictions == -1
        attacks = data[data['attack']]
        log_user_activity("system", f"Detected {len(attacks)} CPS attacks")
        return data, attacks
    except Exception as e:
        logger.error(f"CPS attack simulation error: {str(e)}")
        st.error(f"CPS attack simulation error: {str(e)}")
        return data, pd.DataFrame()

# Retrain model
def retrain_model(df, label_encoders, le_class):
    try:
        df_processed, label_encoders, le_class = preprocess_data(df, label_encoders, le_class, is_train=True)
        if df_processed is None:
            return None, None, None, None
        X = df_processed.drop(columns=['class'], errors='ignore')
        y = df_processed['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = XGBClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, 'idps_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        joblib.dump(le_class, 'le_class.pkl')
        log_user_activity("system", "Model retrained")
        return model, scaler, label_encoders, le_class
    except Exception as e:
        logger.error(f"Model retraining error: {str(e)}")
        st.error(f"Model retraining error: {str(e)}")
        return None, None, None, None

# Intrusion prediction
def predict_traffic(input_data, model, scaler, label_encoders, le_class, threshold=0.5):
    try:
        input_data = input_data.copy()
        
        for col in CATEGORICAL_COLS:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)
                unseen_mask = ~input_data[col].isin(label_encoders[col].classes_)
                input_data.loc[unseen_mask, col] = 'unknown'
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        
        input_data = input_data.drop(columns=LOW_IMPORTANCE_FEATURES, errors='ignore')
        
        expected_features = [col for col in NSL_KDD_COLUMNS if col not in LOW_IMPORTANCE_FEATURES + ['class']]
        for col in expected_features:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
            else:
                input_data[col] = 0
        
        input_data = input_data[expected_features]
        input_data_scaled = scaler.transform(input_data)
        pred_prob = model.predict_proba(input_data_scaled)[:, 1]
        prediction = (pred_prob >= threshold).astype(int)
        prediction_label = le_class.inverse_transform(prediction)[0]
        
        return prediction_label, pred_prob[0]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Calculate compliance metrics
def calculate_compliance_metrics(detection_rate, open_ports, alerts, anomalies=0, threats=0):
    scores = {
        'Detection Rate': min(100, detection_rate * 100),
        'Open Ports': max(0, 100 - open_ports * 10),
        'Alert Frequency': max(0, 100 - alerts * 5),
        'Anomaly Detection': max(0, 100 - anomalies * 10),
        'Threat Intelligence': max(0, 100 - threats * 5)
    }
    overall = sum(scores.values()) / len(scores)
    return scores, overall

# PDF report generation
def generate_nama_report(scan_results=None, atc_results=None, compliance_scores=None, anomalies=None, threats=None):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("NAMA Cybersecurity Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = ("This report details cybersecurity findings for NAMA's network infrastructure, "
                       "including network scans, air traffic control monitoring, compliance metrics, "
                       "anomaly detection, and threat intelligence.")
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        if scan_results:
            story.append(Paragraph("Network Scan Results", styles['Heading2']))
            data = [["Port", "Protocol", "State", "Service"]]
            open_ports = [r for r in scan_results if r['state'] == 'open']
            for r in open_ports[:10]:
                data.append([str(r['port']), r['protocol'], r['state'], r['service']])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Paragraph(f"Total open ports: {len(open_ports)}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if atc_results:
            story.append(Paragraph("ATC Monitoring Results", styles['Heading2']))
            intrusions = [r for r in atc_results if r.get('prediction') != 'normal']
            data = [["Timestamp", "Airport Code", "Prediction", "Confidence"]]
            for r in intrusions[:10]:
                data.append([str(r['timestamp']), r['airport_code'], r['prediction'], f"{r['confidence']:.2%}"])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Paragraph(f"Total intrusions detected: {len(intrusions)}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if anomalies is not None and not anomalies.empty:
            story.append(Paragraph("Air Traffic Anomalies", styles['Heading2']))
            data = [["Timestamp", "Airport Code", "Latitude", "Longitude", "Altitude"]]
            for _, row in anomalies.head(10).iterrows():
                data.append([
                    str(row['timestamp']),
                    row['airport_code'],
                    f"{row['latitude']:.2f}",
                    f"{row['longitude']:.2f}",
                    f"{row['altitude']:.0f}"
                ])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Paragraph(f"Total anomalies detected: {len(anomalies)}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if threats:
            story.append(Paragraph("Cyber Threat Intelligence", styles['Heading2']))
            data = [["Threat Description"]]
            for threat in threats[:10]:
                display_text = threat[:100] + "..." if len(threat) > 100 else threat
                data.append([display_text])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Paragraph(f"Total threats identified: {len(threats)}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if compliance_scores:
            story.append(Paragraph("Compliance Status", styles['Heading2']))
            data = [["Metric", "Score"]]
            for metric, score in compliance_scores.items():
                data.append([metric, f"{score:.1f}%"])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            overall_score = sum(compliance_scores.values()) / len(compliance_scores)
            story.append(Paragraph(f"Overall Compliance Score: {overall_score:.1f}%", styles['Normal']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = (
            "- Implement stricter firewall rules for open ports.\n"
            "- Conduct regular security audits and penetration testing.\n"
            "- Enhance monitoring of air traffic control systems.\n"
            "- Train staff on recognizing insider threats.\n"
            "- Update intrusion detection models with new data."
        )
        story.append(Paragraph(recommendations, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        log_user_activity("system", "Generated NAMA cybersecurity report")
        return buffer
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        st.error(f"Report generation error: {str(e)}")
        return None

# Main Streamlit app
def main():
    setup_user_db()
    apply_wicket_css(st.session_state.theme_mode)
    
    # Load models and encoders
    try:
        model = joblib.load('idps_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        le_class = joblib.load('le_class.pkl')
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        model, scaler, label_encoders, le_class = None, None, {}, None
    
    # Authentication
    if not st.session_state.authenticated:
        st.markdown("""
            <div class="auth-container">
                <h1 class="glitch" style="font-family: 'Poppins', sans-serif; color: #F7FAFC; text-align: center; font-size: 2rem; margin-top: 1rem;">NAMA IDPS</h1>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            with st.form("signin_form"):
                st.markdown('<div class="auth-form">', unsafe_allow_html=True)
                st.markdown('<h2 class="form__title">Sign In</h2>', unsafe_allow_html=True)
                email = st.text_input("Email", key="signin_email", help="Enter your email")
                password = st.text_input("Password", type="password", key="signin_password", help="Enter your password")
                st.markdown('<a href="#" class="auth-link">Forgot your password?</a>', unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", type="primary")
                st.markdown('</div>', unsafe_allow_html=True)
                if submit:
                    if authenticate_user(email, password):
                        try:
                            st.session_state.authenticated = True
                            st.session_state.username = email
                            log_user_activity(email, "Logged in")
                            st.success("Login successful!")
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Session state assignment error: {str(e)}")
                            st.error("An error occurred during login. Please try again.")
                    else:
                        st.markdown("""
                            <script>
                                var container = document.querySelector('.auth-container');
                                container.classList.add('shake');
                                setTimeout(() => container.classList.remove('shake'), 400);
                            </script>
                        """, unsafe_allow_html=True)
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("signup_form"):
                st.markdown('<div class="auth-form">', unsafe_allow_html=True)
                st.markdown('<h2 class="form__title">Sign Up</h2>', unsafe_allow_html=True)
                username = st.text_input("User", key="signup_username", help="Enter your username")
                email = st.text_input("Email", key="signup_email", help="Enter your email")
                password = st.text_input("Password", type="password", key="signup_password", help="Enter your password")
                submit = st.form_submit_button("Sign Up", type="primary")
                st.markdown('</div>', unsafe_allow_html=True)
                if submit:
                    if register_user(username, password):
                        st.success("User registered successfully! Please sign in.")
                        log_user_activity(username, "Registered")
                    else:
                        st.error("Registration failed: Username already exists or bcrypt is missing")
        return
    
    # Main dashboard
    st.sidebar.image("https://via.placeholder.com/120x60.png?text=NAMA+IDPS", use_column_width=True)
    st.sidebar.selectbox("Theme", ["Dark"], index=0)
    st.title("NAMA Intrusion Detection and Prevention System")
    st.markdown(f"Welcome, {st.session_state.username} | [Logout](#)", unsafe_allow_html=True)
    
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        log_user_activity(st.session_state.username or "unknown", "Logged out")
        st.rerun()
    
    menu = st.sidebar.selectbox(
        "Menu",
        ["Dashboard", "Network Scan", "ATC Monitoring", "Threat Intelligence", "Predictive Maintenance", "Reports", "Model Training"]
    )
    
    if menu == "Dashboard":
        st.header("System Dashboard")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Compliance Metrics")
            detection_rate = len([a for a in st.session_state.alert_log if a['severity'] == 'high']) / max(1, len(st.session_state.alert_log))
            open_ports = st.session_state.compliance_metrics.get('open_ports', 0)
            alerts = len(st.session_state.alert_log)
            scores, overall = calculate_compliance_metrics(detection_rate, open_ports, alerts)
            st.session_state.compliance_metrics = scores
            fig = px.bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                title="Compliance Scores",
                labels={'x': 'Metric', 'y': 'Score (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Overall Compliance", f"{overall:.1f}%")
        
        with col2:
            st.subheader("Recent Alerts")
            if st.session_state.alert_log:
                alert_df = pd.DataFrame(st.session_state.alert_log[-5:])
                st.dataframe(alert_df[['timestamp', 'type', 'severity']])
            else:
                st.info("No alerts recorded.")
        
        st.subheader("Network Activity")
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            fig = px.line(
                history_df,
                x='timestamp',
                y='confidence',
                color='prediction',
                title="Intrusion Detection History"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif menu == "Network Scan":
        st.header("Network Vulnerability Scan")
        scan_type = st.selectbox("Scan Type", ["TCP SYN", "TCP Connect", "UDP"])
        target = st.text_input("Target IP/Hostname", "192.168.1.1")
        port_range = st.text_input("Port Range", "1-1000")
        custom_args = st.text_input("Custom NMAP Arguments", "-Pn")
        
        if st.button("Run Scan"):
            def scan_callback(host, result):
                st.session_state.scan_results = result.get('scan', {})
            
            if NMAP_AVAILABLE:
                scan_results = run_nmap_scan(target, scan_type, port_range, custom_args, scan_callback)
            else:
                scan_results = simulate_nmap_scan(target, scan_type, port_range)
            
            if scan_results:
                st.session_state.compliance_metrics['open_ports'] = len([r for r in scan_results if r['state'] == 'open'])
                st.dataframe(pd.DataFrame(scan_results))
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Network Scan',
                    'severity': 'medium',
                    'details': f"Scanned {target}, found {st.session_state.compliance_metrics['open_ports']} open ports"
                })
    
    elif menu == "ATC Monitoring":
        st.header("Air Traffic Control Monitoring")
        data_source = st.radio("Data Source", ["Simulated", "Real ADS-B"])
        num_samples = st.slider("Number of Samples", 1, 100, 10)
        
        if st.button("Analyze Traffic"):
            if data_source == "Real ADS-B":
                traffic_data = fetch_adsb_data(num_samples)
            else:
                traffic_data = simulate_aviation_traffic(num_samples)
            
            if traffic_data:
                traffic_df = pd.DataFrame(traffic_data)
                traffic_df, anomalies = detect_air_traffic_anomalies(traffic_df)
                
                results = []
                for _, row in traffic_df.iterrows():
                    prediction, confidence = predict_traffic(
                        pd.DataFrame([row]), model, scaler, label_encoders, le_class
                    )
                    results.append({
                        'timestamp': row['timestamp'],
                        'airport_code': row['airport_code'],
                        'prediction': prediction,
                        'confidence': confidence
                    })
                
                st.session_state.analysis_history.extend(results)
                st.session_state.atc_results = results
                
                st.subheader("Traffic Analysis")
                st.dataframe(pd.DataFrame(results))
                
                if not anomalies.empty:
                    st.subheader("Detected Anomalies")
                    st.dataframe(anomalies[['timestamp', 'airport_code', 'latitude', 'longitude', 'altitude']])
                    st.session_state.alert_log.append({
                        'timestamp': datetime.now(),
                        'type': 'ATC Anomaly',
                        'severity': 'high',
                        'details': f"Detected {len(anomalies)} air traffic anomalies"
                    })
                
                # Visualization
                fig = px.scatter(
                    traffic_df,
                    x='longitude',
                    y='latitude',
                    size='altitude',
                    color='anomaly',
                    hover_data=['airport_code', 'velocity'],
                    title="Air Traffic Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif menu == "Threat Intelligence":
        st.header("Cyber Threat Intelligence")
        threat_feed = st.text_area("Enter Threat Feed (one per line)", "Suspicious IP detected\nMalware signature found")
        if st.button("Analyze Threats"):
            threats = analyze_threat_feeds(threat_feed.split('\n'))
            if threats:
                st.session_state.threats = threats
                st.write("Detected Threats:")
                for threat in threats:
                    st.write(f"- {threat}")
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Threat Intelligence',
                    'severity': 'high',
                    'details': f"Detected {len(threats)} cyber threats"
                })
    
    elif menu == "Predictive Maintenance":
        st.header("Predictive Maintenance")
        equipment_data = st.file_uploader("Upload Equipment Data (CSV)", type="csv")
        if equipment_data and st.button("Analyze Equipment"):
            equip_df = pd.read_csv(equipment_data)
            equip_df, anomalies = predict_maintenance_anomalies(equip_df)
            st.dataframe(anomalies)
            if not anomalies.empty:
                st.session_state.alert_log.append({
                    'timestamp': datetime.now(),
                    'type': 'Equipment Anomaly',
                    'severity': 'medium',
                    'details': f"Detected {len(anomalies)} equipment anomalies"
                })
    
    elif menu == "Reports":
        st.header("Generate Report")
        if st.button("Generate Cybersecurity Report"):
            report_buffer = generate_nama_report(
                scan_results=st.session_state.get('scan_results', []),
                atc_results=st.session_state.get('atc_results', []),
                compliance_scores=st.session_state.compliance_metrics,
                anomalies=pd.DataFrame(st.session_state.get('atc_results', [])).query("prediction != 'normal'"),
                threats=st.session_state.get('threats', [])
            )
            if report_buffer:
                st.download_button(
                    label="Download Report",
                    data=report_buffer,
                    file_name="nama_cybersecurity_report.pdf",
                    mime="application/pdf"
                )
    
    elif menu == "Model Training":
        st.header("Model Training")
        training_data = st.file_uploader("Upload Training Data (CSV)", type="csv")
        if training_data and st.button("Retrain Model"):
            df = pd.read_csv(training_data, names=NSL_KDD_COLUMNS, low_memory=False)
            model, scaler, label_encoders, le_class = retrain_model(df, label_encoders, le_class)
            if model:
                st.success("Model retrained successfully")
    
    # Display alerts
    if st.session_state.alert_log:
        st.sidebar.subheader("Recent Alerts")
        for alert in st.session_state.alert_log[-3:]:
            st.sidebar.markdown(f"**{alert['timestamp']}**: {alert['type']} ({alert['severity']})")

if __name__ == "__main__":
    main()
