
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

# Logging setup
logging.basicConfig(filename='nama_idps.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (from previous context)
BCRYPT_AVAILABLE = True  # Assume bcrypt is installed after previous fix
NSL_KDD_COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                   'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
                   'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                   'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                   'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
                   'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
                   'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                   'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class']
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
LOW_IMPORTANCE_FEATURES = ['num_outbound_cmds', 'is_host_login']

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'le_class' not in st.session_state:
    st.session_state.le_class = None
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'threats' not in st.session_state:
    st.session_state.threats = []
if 'flight_conflicts' not in st.session_state:
    st.session_state.flight_conflicts = []
if 'drone_results' not in st.session_state:
    st.session_state.drone_results = []
if 'radar_data' not in st.session_state:
    st.session_state.radar_data = None
if 'atc_results' not in st.session_state:
    st.session_state.atc_results = []
if 'atc_anomalies' not in st.session_state:
    st.session_state.atc_anomalies = pd.DataFrame()
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []

def apply_wicket_css():
    st.markdown("""
        <style>
        .stApp {
            background: url("https://raw.githubusercontent.com/J4yd33n/IDPS-with-ML/main/images/aeroplane.jpg") no-repeat center center fixed;
            background-size: cover;
            color: #E6E6FA;
        }
        .sidebar .sidebar-content {
            background-color: rgba(30, 30, 30, 0.8);
            color: #E6E6FA;
        }
        .card {
            background-color: rgba(30, 30, 30, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FFFF;
            text-shadow: 0 0 10px #00FFFF;
        }
        .stButton > button {
            background: linear-gradient(90deg, #0367A6, #008997);
            color: #E6E6FA;
            border: none;
            border-radius: 20px;
            padding: 10px 20px;
            font-weight: bold;
            text-transform: uppercase;
            transition: transform 80ms ease-in;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px #00FFFF;
        }
        .stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.1);
            color: #E6E6FA;
            border: 1px solid #00FFFF;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

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
    if BCRYPT_AVAILABLE:
        default_password = 'admin'
        hashed = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", ('nama', hashed))
        conn.commit()
        logger.info("Default user 'nama' created or verified")
    conn.close()

def register_user(username, password):
    if not BCRYPT_AVAILABLE:
        logger.error("Registration failed: bcrypt module is missing")
        return False
    try:
        conn = sqlite3.connect('nama_users.db')
        c = conn.cursor()
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Registration failed: Username {username} already exists")
        return False
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return False

def authenticate_user(username, password):
    if not BCRYPT_AVAILABLE:
        logger.error("Authentication disabled: bcrypt module is missing")
        return False
    try:
        conn = sqlite3.connect('nama_users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        if result:
            stored_password = result[0]
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')
            return bcrypt.checkpw(password.encode('utf-8'), stored_password)
        return False
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return False

def log_user_activity(username, action):
    try:
        conn = sqlite3.connect('nama_users.db')
        c = conn.cursor()
        timestamp = pd.Timestamp.now().isoformat()
        c.execute("INSERT INTO user_activity (username, timestamp, action) VALUES (?, ?, ?)",
                  (username, timestamp, action))
        conn.commit()
        conn.close()
        logger.info(f"User: {username}, Action: {action}")
    except Exception as e:
        logger.error(f"Error logging user activity: {str(e)}")

def preprocess_data(df, label_encoders=None, le_class=None, is_train=True):
    try:
        df = df.copy()
        if label_encoders is None:
            label_encoders = {}
        if le_class is None and is_train:
            le_class = LabelEncoder()
        for col in NSL_KDD_COLUMNS:
            if col not in df.columns and col != 'class':
                df[col] = 0
            elif col in CATEGORICAL_COLS and col not in df.columns:
                df[col] = 'missing'
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
                if col not in label_encoders:
                    logger.error(f"Label encoder for {col} not found during inference")
                    return None, label_encoders, le_class
                unseen_mask = ~df[col].isin(label_encoders[col].classes_)
                df.loc[unseen_mask, col] = 'unknown'
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                df[col] = label_encoders[col].transform(df[col])
        if 'class' in df.columns:
            df['class'] = df['class'].astype(str)
            if is_train:
                df['class'] = le_class.fit_transform(df['class'])
            else:
                if le_class is None:
                    logger.error("Class label encoder not provided for inference")
                    return None, label_encoders, le_class
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

def train_model(data):
    # Placeholder for model training (from previous context)
    try:
        processed_data, label_encoders, le_class = preprocess_data(data, is_train=True)
        if processed_data is None:
            return None, None, None, None
        # Simulate model training
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = processed_data.drop(columns=['class'], errors='ignore')
        X_scaled = scaler.fit_transform(X)
        return "model_placeholder", scaler, label_encoders, le_class
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        return None, None, None, None

# Placeholder functions for other features
def periodic_adsb_fetch(interval): pass
def periodic_radar_update(interval): pass
def periodic_nmap_scan(target, scan_type, port_range): pass
def periodic_drone_detection(): pass
def periodic_threat_fetch(source): pass
def periodic_compliance_check(): pass
def display_radar(radar_data): return None
def display_drone_data(): pass
def display_threat_intelligence(): pass
def display_compliance_metrics(): pass
def download_report(): pass
def run_nmap_scan(target, scan_type, port_range, custom_args): return []

def main():
    apply_wicket_css()
    setup_user_db()

    if 'form_type' not in st.session_state:
        st.session_state.form_type = 'signin'

    if not st.session_state.model:
        sample_data = pd.DataFrame({
            'duration': [0, 100, 200], 'protocol_type': ['tcp', 'udp', 'icmp'], 'service': ['http', 'ftp', 'ssh'],
            'flag': ['SF', 'S0', 'REJ'], 'src_bytes': [100, 500, 1500], 'dst_bytes': [0, 100, 1000],
            'land': [0, 0, 0], 'wrong_fragment': [0, 0, 0], 'urgent': [0, 0, 0], 'hot': [0, 0, 0],
            'num_failed_logins': [0, 0, 0], 'logged_in': [0, 0, 0], 'num_compromised': [0, 0, 0],
            'root_shell': [0, 0, 0], 'su_attempted': [0, 0, 0], 'num_root': [0, 0, 0],
            'num_file_creations': [0, 0, 0], 'num_shells': [0, 0, 0], 'num_access_files': [0, 0, 0],
            'num_outbound_cmds': [0, 0, 0], 'is_host_login': [0, 0, 0], 'is_guest_login': [0, 0, 0],
            'count': [1, 2, 3], 'srv_count': [1, 2, 3], 'serror_rate': [0.0, 0.0, 0.0],
            'srv_serror_rate': [0.0, 0.0, 0.0], 'rerror_rate': [0.0, 0.0, 0.0], 'srv_rerror_rate': [0.0, 0.0, 0.0],
            'same_srv_rate': [1.0, 1.0, 1.0], 'diff_srv_rate': [0.0, 0.0, 0.0], 'srv_diff_host_rate': [0.0, 0.0, 0.0],
            'dst_host_count': [100, 100, 100], 'dst_host_srv_count': [100, 100, 100],
            'dst_host_same_srv_rate': [1.0, 1.0, 1.0], 'dst_host_diff_srv_rate': [0.0, 0.0, 0.0],
            'dst_host_same_src_port_rate': [0.0, 0.0, 0.0], 'dst_host_srv_diff_host_rate': [0.0, 0.0, 0.0],
            'dst_host_serror_rate': [0.0, 0.0, 0.0], 'dst_host_srv_serror_rate': [0.0, 0.0, 0.0],
            'dst_host_rerror_rate': [0.0, 0.0, 0.0], 'dst_host_srv_rerror_rate': [0.0, 0.0, 0.0],
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
                        <input type="text" placeholder="Username" class="input" id="signinUsername" />
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
                    const username = document.getElementById("signinUsername").value;
                    const password = document.getElementById("signinPassword").value;
                    fetch("/_stcore/streamlit_script_run", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            "signin": {
                                "username": username,
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
                        logger.info(f"Session state updated: authenticated={st.session_state.authenticated}, username={st.session_state.username}")
                        log_user_activity(username, "Signed in")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password")
        except Exception as e:
            logger.error(f"Error processing form submission: {str(e)}")
            st.error(f"Error processing form submission: {str(e)}")
    else:
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

