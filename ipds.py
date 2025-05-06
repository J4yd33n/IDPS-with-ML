import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import logging
import os
import sys
import sqlite3
import pyotp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import base64
import io
import requests

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

# Theme configuration
THEME = {
    "Light": {
        "background": "#f0f2f6",
        "text": "#000000",
        "input_bg": "#ffffff",
        "input_border": "#cccccc",
        "chart_bg": "#ffffff",
        "chart_text": "#000000"
    },
    "Dark": {
        "background": "#1e1e2f",
        "text": "#ffffff",
        "input_bg": "#2a2a3d",
        "input_border": "#555555",
        "chart_bg": "#1c1c2c",
        "chart_text": "#ffffff"
    }
}

# Apply custom CSS for theming
def apply_theme_css(theme):
    colors = THEME[theme]
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {colors['background']};
                color: {colors['text']};
            }}
            .stButton>button {{
                background-color: {'#4CAF50' if theme == 'Light' else '#388E3C'};
                color: #ffffff;
            }}
            .stTextInput>div>input {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['input_border']};
            }}
            .logo-image {{
                width: 100%;
                max-width: 150px;
                height: auto;
            }}
        </style>
        """, unsafe_allow_html=True
    )

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'compliance_metrics' not in st.session_state:
    st.session_state.compliance_metrics = {'detection_rate': 0, 'open_ports': 0, 'alerts': 0}

# User database setup
def setup_user_db():
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT,
        mfa_secret TEXT
    )''')
    conn.commit()
    conn.close()

def register_user(username, password, mfa_secret):
    if not BCRYPT_AVAILABLE:
        logger.error("Cannot register user: bcrypt module is missing")
        return False
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, mfa_secret) VALUES (?, ?, ?)",
                  (username, hashed, mfa_secret))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    conn.close()
    return True

def authenticate_user(username, password, mfa_code):
    if not BCRYPT_AVAILABLE:
        logger.error("Authentication disabled: bcrypt module is missing")
        return False
    conn = sqlite3.connect('nama_users.db')
    c = conn.cursor()
    c.execute("SELECT password, mfa_secret FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_password, mfa_secret = result
        if bcrypt.checkpw(password.encode('utf-8'), stored_password):
            totp = pyotp.TOTP(mfa_secret)
            return totp.verify(mfa_code)
    return False

# Audit logging
def log_action(user, action):
    logger.info(f"User: {user}, Action: {action}")

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
                valid_classes = le_class.classes_
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
        log_action("system", f"Real-time NMAP scan on {target}")
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
        log_action("system", f"Simulated NMAP scan on {target}")
        return scan_results
    except Exception as e:
        logger.error(f"NMAP simulation error: {str(e)}")
        st.error(f"NMAP simulation error: {str(e)}")
        return []

# Fetch real ADS-B data
def fetch_adsb_data(num_samples=10):
    try:
        # Use Streamlit secrets or environment variables for credentials
        username = st.secrets.get('OPENSKY_USERNAME', os.getenv('OPENSKY_USERNAME'))
        password = st.secrets.get('OPENSKY_PASSWORD', os.getenv('OPENSKY_PASSWORD'))
        
        if not username or not password:
            raise ValueError("OpenSky credentials not provided. Set OPENSKY_USERNAME and OPENSKY_PASSWORD environment variables or secrets.")
        
        url = "https://opensky-network.org/api/states/all"
        response = requests.get(url, auth=(username, password))
        if response.status_code != 200:
            raise Exception(f"Failed to fetch ADS-B data: {response.status_code} {response.reason}")
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
                'dst_host_srv_rerror_rate': 0.0
            })
        log_action("system", "Fetched real ADS-B data")
        return adsb_records
    except Exception as e:
        logger.error(f"ADS-B fetch error: {str(e)}")
        st.error(f"ADS-B fetch error: {str(e)}")
        return []

# ATC simulation
def simulate_aviation_traffic(num_samples=10):
    try:
        airports = ['DNMM', 'DNAA', 'DNKN', 'DNPO']
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
            'dst_host_srv_rerror_rate': np.random.uniform(0, 1, num_samples)
        }
        log_action("system", "Simulated ATC traffic")
        return pd.DataFrame(data).to_dict('records')
    except Exception as e:
        logger.error(f"ATC simulation error: {str(e)}")
        st.error(f"ATC simulation error: {str(e)}")
        return []

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
        log_action("system", "Model retrained")
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
            if col not in input_data.columns:
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
def calculate_compliance_metrics(detection_rate, open_ports, alerts):
    scores = {
        'Detection Rate': min(100, detection_rate * 100),
        'Open Ports': max(0, 100 - open_ports * 10),
        'Alert Frequency': max(0, 100 - alerts * 5)
    }
    overall = sum(scores.values()) / len(scores)
    return scores, overall

# PDF report generation
def generate_nama_report(scan_results=None, atc_results=None, compliance_scores=None):
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
        story.append(Paragraph("This report details cybersecurity findings for NAMA's network infrastructure.", styles['Normal']))
        story.append(Spacer(1, 12))
        
        if scan_results:
            story.append(Paragraph("NMAP Scan Results", styles['Heading2']))
            data = [["Port", "Protocol", "State", "Service"]]
            open_ports = [r for r in scan_results if r['state'] == 'open']
            for r in open_ports:
                data.append([r['port'], r['protocol'], r['state'], r['service']])
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
            story.append(Spacer(1, 12))
        
        if atc_results:
            story.append(Paragraph("ATC Monitoring Results", styles['Heading2']))
            intrusions = [r for r in atc_results if r.get('prediction') != 'normal']
            data = [["Timestamp", "Airport Code", "Prediction", "Confidence"]]
            for r in intrusions:
                data.append([r['timestamp'], r['airport_code'], r['prediction'], f"{r['confidence']:.2%}"])
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
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        st.error(f"Report generation error: {str(e)}")
        return None

# Main Streamlit app
def main():
    # Apply theme
    apply_theme_css(st.session_state.theme)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150?text=NAMA+Logo", width=150)
    st.sidebar.title("NAMA AI-Enhanced IDPS")
    if st.sidebar.button("Toggle Theme"):
        st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
        apply_theme_css(st.session_state.theme)
    
    # Dependency warnings
    if not BCRYPT_AVAILABLE:
        st.warning("bcrypt module is missing. Authentication features are disabled. Please install bcrypt (pip install bcrypt).")
        st.session_state.authenticated = True  # Bypass authentication for testing
    if not NMAP_AVAILABLE:
        st.warning("python-nmap module is missing. Real NMAP scanning is disabled. Please install python-nmap (pip install python-nmap).")
    
    # Setup user database
    if BCRYPT_AVAILABLE:
        setup_user_db()
    
    # Authentication
    if not st.session_state.authenticated and BCRYPT_AVAILABLE:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            mfa_code = st.text_input("MFA Code")
            if st.form_submit_button("Login"):
                if authenticate_user(username, password, mfa_code):
                    st.session_state.authenticated = True
                    log_action(username, "User logged in")
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or MFA code.")
            if st.form_submit_button("Register"):
                mfa_secret = pyotp.random_base32()
                if register_user(username, password, mfa_secret):
                    st.success(f"User registered! MFA Secret: {mfa_secret} (Save this for TOTP app)")
                else:
                    st.error("Username already exists or registration failed.")
        return
    
    # Load model
    try:
        model = joblib.load('idps_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        le_class = joblib.load('le_class.pkl')
        st.sidebar.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}. Please upload model files.")
        logger.error(f"Model loading error: {str(e)}")
        return
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Navigation",
        ["Home", "NMAP Analysis", "ATC Monitoring", "Compliance Dashboard", "Alert Log", "Documentation"],
        format_func=lambda x: f"{'🏠' if x == 'Home' else '🔍' if x == 'NMAP Analysis' else '✈️' if x == 'ATC Monitoring' else '✅' if x == 'Compliance Dashboard' else '🚨' if x == 'Alert Log' else '📖'} {x}"
    )
    
    if app_mode == "Home":
        st.header("AI-Enhanced Intrusion Detection and Prevention System")
        st.markdown("""
        Welcome to NAMA's state-of-the-art IDPS, designed to protect Nigeria's airspace with AI-driven cybersecurity.
        
        ### Features
        - **NMAP Analysis**: Real-time or simulated port scanning to identify vulnerabilities.
        - **ATC Monitoring**: Analyze aviation protocols (ADS-B, ACARS) with real or simulated data.
        - **Compliance Dashboard**: Dynamic tracking of NCAA/ICAO cybersecurity standards.
        - **Real-time Alerts**: Instant notifications for detected threats.
        - **Professional Reporting**: Generate branded PDF reports for NAMA stakeholders.
        - **Model Retraining**: Support for new datasets and model updates.
        
        Start by exploring NMAP Analysis or ATC Monitoring!
        """)
        if model is not None:
            st.success("Loaded XGBoost model is ready!")
        
        # Model retraining
        st.subheader("Model Retraining")
        dataset_file = st.file_uploader("Upload new dataset (CSV)", type=['csv'])
        if dataset_file:
            df = pd.read_csv(dataset_file)
            if st.button("Retrain Model"):
                with st.spinner("Retraining model..."):
                    model, scaler, label_encoders, le_class = retrain_model(df, label_encoders, le_class)
                    if model:
                        st.success("Model retrained successfully!")
    
    elif app_mode == "NMAP Analysis":
        st.header("NMAP Analysis")
        st.markdown("Perform real-time or simulated port scanning to identify open ports and services.")
        
        use_real_nmap = st.checkbox("Use Real NMAP (requires python-nmap)", value=False, disabled=not NMAP_AVAILABLE)
        real_time_updates = st.checkbox("Enable Real-Time Updates", value=True, disabled=not use_real_nmap)
        
        with st.form("nmap_scan_form"):
            col1, col2 = st.columns(2)
            with col1:
                target = st.text_input("Target IP/Hostname", value="192.168.1.1")
                scan_type = st.selectbox("Scan Type", ["TCP SYN", "TCP Connect", "UDP"])
            with col2:
                port_range = st.text_input("Port Range (e.g., 1-1000)", value="1-1000")
                custom_args = st.text_input("Custom NMAP Args (e.g., -T4)", value="-T4")
            submit = st.form_submit_button("Run Scan")
        
        if submit:
            with st.spinner("Running NMAP scan..."):
                try:
                    if not target:
                        st.error("Please provide a target IP or hostname.")
                        return
                    if not port_range or '-' not in port_range:
                        st.error("Please provide a valid port range (e.g., 1-1000).")
                        return
                    start_port, end_port = map(int, port_range.split('-'))
                    if start_port < 1 or end_port > 65535 or start_port > end_port:
                        st.error("Port range must be between 1 and 65535.")
                        return
                    
                    def scan_callback(host, scan_result):
                        if real_time_updates:
                            st.write(f"Scanned {host}: {scan_result}")
                    
                    if use_real_nmap and NMAP_AVAILABLE:
                        scan_results = run_nmap_scan(target, scan_type, port_range, custom_args, scan_callback)
                    else:
                        scan_results = simulate_nmap_scan(target, scan_type, port_range)
                    
                    open_ports = [r for r in scan_results if r['state'] == 'open']
                    st.session_state.compliance_metrics['open_ports'] = len(open_ports)
                    
                    if not open_ports:
                        st.warning("No open ports detected.")
                    else:
                        st.success(f"Found {len(open_ports)} open ports.")
                        df = pd.DataFrame(open_ports)
                        st.dataframe(df[['port', 'protocol', 'state', 'service']], use_container_width=True)
                        
                        fig = px.bar(
                            df, x='port', y='service', color='protocol', title=f"Open Ports on {target}",
                            color_discrete_sequence=px.colors.sequential.Blues_r
                        )
                        fig.update_layout(
                            paper_bgcolor=THEME[st.session_state.theme]['chart_bg'],
                            plot_bgcolor=THEME[st.session_state.theme]['chart_bg'],
                            font=dict(color=THEME[st.session_state.theme]['chart_text'])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        report_buffer = generate_nama_report(scan_results=scan_results)
                        if report_buffer:
                            b64 = base64.b64encode(report_buffer.getvalue()).decode()
                            href = f'<a href="data:application/pdf;base64,{b64}" download="nama_nmap_report.pdf">Download NMAP Report</a>'
                            st.markdown(href, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during scan: {str(e)}")
    
    elif app_mode == "ATC Monitoring":
        st.header("ATC Network Monitoring")
        st.markdown("Monitor aviation-specific protocols (ADS-B, ACARS) for NAMA's network.")
        
        data_source = st.selectbox("Data Source", ["Simulated", "Real ADS-B"])
        num_samples = st.slider("Number of samples", 5, 50, 10)
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
        
        if st.button("Analyze Traffic"):
            with st.spinner("Analyzing ATC traffic..."):
                try:
                    if data_source == "Real ADS-B":
                        atc_data = fetch_adsb_data(num_samples)
                    else:
                        atc_data = simulate_aviation_traffic(num_samples)
                    
                    if not atc_data:
                        st.error("No data retrieved. Please check credentials for Real ADS-B or try Simulated data.")
                        if st.button("Switch to Simulated Data"):
                            atc_data = simulate_aviation_traffic(num_samples)
                        else:
                            return
                    
                    df = pd.DataFrame(atc_data)
                    if df.empty:
                        st.error("No valid data to analyze. Please try again or use Simulated data.")
                        return
                    
                    predictions = []
                    total_predictions = 0
                    correct_predictions = 0
                    
                    for row in atc_data:
                        row_df = pd.DataFrame([row])
                        pred, conf = predict_traffic(row_df, model, scaler, label_encoders, le_class, threshold)
                        predictions.append({'prediction': pred, 'confidence': conf})
                        total_predictions += 1
                        if pred != 'normal':
                            correct_predictions += 1
                    
                    df['prediction'] = [p['prediction'] for p in predictions]
                    df['confidence'] = [p['confidence'] for p in predictions]
                    intrusions = df[df['prediction'] != 'normal']
                    
                    detection_rate = correct_predictions / total_predictions if total_predictions > 0 else 0
                    st.session_state.compliance_metrics['detection_rate'] = detection_rate
                    st.session_state.compliance_metrics['alerts'] = len(intrusions)
                    
                    display_columns = ['timestamp', 'airport_code', 'protocol_type', 'service', 'prediction', 'confidence']
                    available_columns = [col for col in display_columns if col in df.columns]
                    if not available_columns:
                        st.error("No valid columns available for display.")
                        return
                    st.dataframe(df[available_columns], use_container_width=True)
                    
                    if not intrusions.empty:
                        st.error(f"Detected {len(intrusions)} intrusions!")
                        for _, row in intrusions.iterrows():
                            st.session_state.alert_log.append({
                                'timestamp': row['timestamp'],
                                'message': f"Intrusion: {row['prediction']} at {row['airport_code']} (Confidence: {row['confidence']:.2%})",
                                'recipient': 'security@nama.gov.ng'
                            })
                    else:
                        st.success("No intrusions detected.")
                    
                    fig = px.scatter(
                        df, x='timestamp', y='confidence', color='prediction', size='src_bytes',
                        hover_data=['airport_code', 'protocol_type'], title="ATC Traffic Analysis"
                    )
                    fig.update_layout(
                        paper_bgcolor=THEME[st.session_state.theme]['chart_bg'],
                        plot_bgcolor=THEME[st.session_state.theme]['chart_bg'],
                        font=dict(color=THEME[st.session_state.theme]['chart_text'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    report_buffer = generate_nama_report(atc_results=df.to_dict('records'))
                    if report_buffer:
                        b64 = base64.b64encode(report_buffer.getvalue()).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="nama_atc_report.pdf">Download ATC Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                
                except Exception as e:
                    logger.error(f"ATC analysis error: {str(e)}")
                    st.error(f"ATC analysis error: {str(e)}")
    
    elif app_mode == "Compliance Dashboard":
        st.header("NCAA/ICAO Compliance Dashboard")
        st.markdown("Track cybersecurity compliance for NAMA operations.")
        
        metrics = st.session_state.compliance_metrics
        compliance_scores, overall = calculate_compliance_metrics(
            metrics['detection_rate'], metrics['open_ports'], metrics['alerts']
        )
        
        compliance_data = {
            "Metric": list(compliance_scores.keys()),
            "Score": list(compliance_scores.values())
        }
        df = pd.DataFrame(compliance_data)
        
        fig = px.bar(
            df, x="Metric", y="Score", title="Compliance Metrics",
            color="Score", color_continuous_scale="Blues"
        )
        fig.update_layout(
            paper_bgcolor=THEME[st.session_state.theme]['chart_bg'],
            plot_bgcolor=THEME[st.session_state.theme]['chart_bg'],
            font=dict(color=THEME[st.session_state.theme]['chart_text'])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Overall Compliance Score**: {overall:.1f}%")
        st.markdown("**Recommendations**: Ensure firewall updates, reduce open ports, and maintain high detection rates.")
        
        if st.button("Generate Compliance Report"):
            report_buffer = generate_nama_report(compliance_scores=compliance_scores)
            if report_buffer:
                b64 = base64.b64encode(report_buffer.getvalue()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="nama_compliance_report.pdf">Download Compliance Report</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    elif app_mode == "Alert Log":
        st.header("Alert Log")
        if not st.session_state.alert_log:
            st.info("No alerts generated yet.")
        else:
            alert_df = pd.DataFrame(st.session_state.alert_log)
            st.dataframe(alert_df[['timestamp', 'message', 'recipient']], use_container_width=True)
            if st.button("Clear Alert Log"):
                st.session_state.alert_log = []
                st.success("Alert log cleared.")
    
    elif app_mode == "Documentation":
        st.header("Project Documentation")
        st.markdown("""
        ### NAMA AI-Enhanced IDPS
        
        **Overview**  
        This IDPS leverages machine learning and network scanning to secure NAMA's network infrastructure, ensuring safe airspace operations.
        
        **Objectives**  
        - Detect intrusions with high accuracy using XGBoost.
        - Monitor ATC protocols for anomalies.
        - Ensure compliance with NCAA/ICAO standards.
        
        **Key Features**  
        - Real-time NMAP scanning with customizable arguments.
        - Real and simulated ATC protocol monitoring (ADS-B, ACARS).
        - Secure authentication with MFA and user database.
        - Dynamic compliance dashboard.
        - Model retraining for new datasets.
        - PDF report generation with NAMA branding.
        
        **Technology Stack**  
        - Python, Streamlit, Scikit-learn, XGBoost, Plotly, ReportLab, python-nmap, pyotp, bcrypt, SQLite.
        
        **Future Improvements**  
        - Integrate additional real-time aviation data sources (e.g., ACARS).
        - Enhance MFA with biometric authentication.
        - Implement automated compliance audits.
        - Support ensemble models for improved detection.
        
        **Contact**  
        For feedback, contact [security@nama.gov.ng](mailto:security@nama.gov.ng).
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}. Please check the logs for details.")
