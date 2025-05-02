import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time
import os
from datetime import datetime, timedelta
import io
import requests
import base64
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# NSL-KDD columns (corrected to include all features)
nsl_kdd_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
]

low_importance_features = [
    'num_outbound_cmds', 'is_host_login', 'su_attempted', 'urgent', 'land',
    'num_access_files', 'num_shells', 'root_shell', 'num_failed_logins',
    'num_file_creations', 'num_root'
]

categorical_cols = ['protocol_type', 'service', 'flag']

# Initialize model variables
model = None
scaler = None
label_encoders = None
le_class = None
autoencoder = None
model_type = None

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []
if 'connection_tracker' not in st.session_state:
    st.session_state.connection_tracker = {}
if 'xai_api_key' not in st.session_state:
    st.session_state.xai_api_key = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

# Custom CSS for professional look with improved dark mode readability
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e2f;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: #ffffff;
        border-radius: 5px;
        border: 1px solid #388E3C;
    }
    .stTextInput>div>input {
        background-color: #ffffff;
        color: #000000;
        border-radius: 5px;
        border: 1px solid #cccccc;
    }
    .stSelectbox, .stSlider, .stRadio, .stCheckbox {
        color: #000000;
    }
    .stMarkdown, .stDataFrame, .stTable {
        color: #000000;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #ffffff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Theme toggle with improved dark mode contrast
def toggle_theme():
    st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {'#1e1e2f' if st.session_state.theme == 'Dark' else '#f0f2f6'};
                color: {'#ffffff' if st.session_state.theme == 'Dark' else '#000000'};
            }}
            .stMarkdown, .stDataFrame, .stTable, .stSelectbox, .stSlider, .stRadio, .stCheckbox {{
                color: {'#ffffff' if st.session_state.theme == 'Dark' else '#000000'};
            }}
            .stTextInput>div>input {{
                background-color: {'#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff'};
                color: {'#ffffff' if st.session_state.theme == 'Dark' else '#000000'};
                border: 1px solid {'#555555' if st.session_state.theme == 'Dark' else '#cccccc'};
            }}
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
                color: {'#ffffff' if st.session_state.theme == 'Dark' else '#000000'};
            }}
            .stPlotlyChart {{
                background-color: {'#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff'};
            }}
        </style>
        """, unsafe_allow_html=True
    )

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('idps_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        le_class = joblib.load('le_class.pkl')
        autoencoder = joblib.load('autoencoder.pkl') if os.path.exists('autoencoder.pkl') else None
        model_type = joblib.load('model_type.pkl') if os.path.exists('model_type.pkl') else 'XGBoost'
        st.success("Model and preprocessing objects loaded successfully.")
        return model, scaler, label_encoders, le_class, autoencoder, model_type
    except Exception as e:
        st.error(f"Failed to load model or preprocessing objects: {str(e)}")
        return None, None, None, None, None, 'XGBoost'

model, scaler, label_encoders, le_class, autoencoder, model_type = load_model()

def preprocess_data(df, label_encoders, le_class, is_train=True):
    df = df.copy()
    
    df.fillna({'protocol_type': 'missing', 'service': 'missing', 'flag': 'missing'}, inplace=True)
    df.fillna(0, inplace=True)
    
    numeric_cols = [
        col for col in nsl_kdd_columns 
        if col not in categorical_cols + ['class'] + low_importance_features
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    for col in categorical_cols:
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
    
    df = df.drop(columns=[col for col in low_importance_features if col in df.columns], errors='ignore')
    
    return df, label_encoders, le_class

def train_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    model.fit(X_train_reshaped, y_train, epochs=5, batch_size=8, validation_data=(X_test_reshaped, y_test), verbose=0)
    return model

def train_autoencoder(X_train, epochs=10):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu')(input_layer)
    encoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(input_dim, activation='linear')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=8, verbose=0)
    return autoencoder

def detect_anomaly(input_data_scaled, autoencoder, threshold=0.1):
    reconstructions = autoencoder.predict(input_data_scaled)
    mse = np.mean(np.square(input_data_scaled - reconstructions), axis=1)
    return mse > threshold

def explain_threat(prediction, confidence, src_ip, xai_api_key=None, openai_api_key=None):
    prompt = f"Explain the network intrusion type '{prediction}' with confidence {confidence:.2%} from source IP {src_ip}. Suggest mitigation actions."
    
    if xai_api_key:
        try:
            response = requests.post(
                'https://api.x.ai/v1/completions',
                headers={'Authorization': f'Bearer {xai_api_key}', 'Content-Type': 'application/json'},
                json={'model': 'grok-beta', 'prompt': prompt, 'max_tokens': 200}
            )
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('text', 'Explanation unavailable')
        except Exception as e:
            st.warning(f"xAI API error: {str(e)}. Trying OpenAI API if provided.")
    
    if openai_api_key:
        try:
            response = requests.post(
                'https://api.openai.com/v1/completions',
                headers={'Authorization': f'Bearer {openai_api_key}', 'Content-Type': 'application/json'},
                json={'model': 'text-davinci-003', 'prompt': prompt, 'max_tokens': 200}
            )
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('text', 'Explanation unavailable')
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
    
    return "No valid API key provided for threat explanation."

def predict_traffic(input_data, threshold=0.5):
    global model, scaler, label_encoders, le_class, model_type
    
    if model is None or scaler is None or label_encoders is None or le_class is None:
        st.error("Model or preprocessing components not loaded.")
        return None, None
    
    try:
        input_data = input_data.copy()
        
        for col in categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)
                unseen_mask = ~input_data[col].isin(label_encoders[col].classes_)
                input_data.loc[unseen_mask, col] = 'unknown'
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        
        input_data = input_data.drop(columns=low_importance_features, errors='ignore')
        
        expected_features = [col for col in nsl_kdd_columns if col not in low_importance_features + ['class']]
        
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[expected_features]
        
        input_data_scaled = scaler.transform(input_data)
        
        if model_type == 'LSTM':
            input_data_reshaped = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)
            pred_prob = model.predict(input_data_reshaped, verbose=0)[:, 0]
        else:
            pred_prob = model.predict_proba(input_data_scaled)[:, 1]
        
        prediction = (pred_prob >= threshold).astype(int)
        prediction_label = le_class.inverse_transform(prediction)[0]
        
        return prediction_label, pred_prob[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def simulate_nmap_scan(target, scan_type, port_range):
    # Mock port data for simulation
    common_ports = {
        21: ('ftp', 'tcp'),
        22: ('ssh', 'tcp'),
        23: ('telnet', 'tcp'),
        25: ('smtp', 'tcp'),
        53: ('dns', 'tcp/udp'),
        80: ('http', 'tcp'),
        110: ('pop3', 'tcp'),
        143: ('imap', 'tcp'),
        443: ('https', 'tcp'),
        3306: ('mysql', 'tcp'),
        3389: ('rdp', 'tcp'),
        5432: ('postgresql', 'tcp'),
        137: ('netbios-ns', 'udp'),
        161: ('snmp', 'udp'),
        500: ('ipsec', 'udp')
    }
    
    # Filter ports based on scan type and port range
    start_port, end_port = map(int, port_range.split('-'))
    ports_to_scan = [p for p in common_ports.keys() if start_port <= p <= end_port]
    
    # Simulate open/closed ports
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
    
    return scan_results

def show_nmap_analysis():
    st.header("NMAP Analysis")
    
    st.markdown("""
    Simulate an NMAP port scan to identify open ports and services on a target host.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Enter a target IP/hostname, select scan type, and specify port range to view results.</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Scan Configuration")
    with st.form("nmap_scan_form"):
        col1, col2 = st.columns(2)
        with col1:
            target = st.text_input("Target IP/Hostname", value="192.168.1.1")
            scan_type = st.selectbox("Scan Type", ["TCP SYN", "TCP Connect", "UDP"])
        with col2:
            port_range = st.text_input("Port Range (e.g., 1-1000)", value="1-1000")
            intensity = st.slider("Scan Intensity", 1, 5, 3, help="Higher intensity simulates more thorough scans")
        
        submit = st.form_submit_button("Run Scan")
    
    if submit:
        with st.spinner("Running NMAP simulation..."):
            try:
                # Validate inputs
                if not target:
                    st.error("Please provide a target IP or hostname.")
                    return
                if not port_range or '-' not in port_range:
                    st.error("Please provide a valid port range (e.g., 1-1000).")
                    return
                start_port, end_port = port_range.split('-')
                start_port, end_port = int(start_port), int(end_port)
                if start_port < 1 or end_port > 65535 or start_port > end_port:
                    st.error("Port range must be between 1 and 65535, with start port less than or equal to end port.")
                    return
                
                # Simulate scan
                scan_results = simulate_nmap_scan(target, scan_type, port_range)
                
                # Display results in NMAP-like format
                st.subheader(f"Scan Results for {target}")
                st.markdown(f"""
                **NMAP Simulation**  
                Scan Type: {scan_type}  
                Port Range: {port_range}  
                Intensity: {intensity}  
                Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                """)
                
                # Filter open ports
                open_ports = [r for r in scan_results if r['state'] == 'open']
                if not open_ports:
                    st.warning("No open ports detected.")
                else:
                    st.success(f"Found {len(open_ports)} open ports.")
                    df = pd.DataFrame(open_ports)
                    st.dataframe(
                        df[['port', 'protocol', 'state', 'service']],
                        column_config={
                            'port': st.column_config.NumberColumn("Port"),
                            'protocol': st.column_config.TextColumn("Protocol"),
                            'state': st.column_config.TextColumn("State"),
                            'service': st.column_config.TextColumn("Service")
                        },
                        use_container_width=True
                    )
                
                # Visualize open ports
                if open_ports:
                    fig = px.bar(
                        df,
                        x='port',
                        y='service',
                        color='protocol',
                        title=f"Open Ports on {target}",
                        labels={'port': 'Port Number', 'service': 'Service'},
                        height=400,
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    fig.update_layout(
                        paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        plot_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        font=dict(color='#ffffff'),
                        title_font=dict(color='#ffffff'),
                        xaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff')),
                        yaxis=dict(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff')),
                        legend=dict(font=dict(color='#ffffff'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export results
                st.subheader("Export Results")
                csv_data = pd.DataFrame(scan_results).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"nmap_scan_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during scan simulation: {str(e)}")

def show_historical_analysis():
    st.header("Historical Analysis Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Analyze a PCAP file first.")
        return
    
    st.markdown("""
    Visualize trends and insights from past analyses.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Filter by date to explore intrusion trends and attack distributions.</span>
    </div>
    """, unsafe_allow_html=True)
    
    history_df = pd.DataFrame([
        {
            'Timestamp': h['timestamp'],
            'Filename': h['filename'],
            'Total Packets': h['total_packets'],
            'Intrusions': h['intrusion_count'],
            'Normal': h['total_packets'] - h['intrusion_count']
        }
        for h in st.session_state.analysis_history
    ])
    
    st.subheader("Date Filter")
    min_date = min(history_df['Timestamp']).date()
    max_date = max(history_df['Timestamp']).date()
    start_date, end_date = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    filtered_df = history_df[
        (history_df['Timestamp'].dt.date >= start_date) &
        (history_df['Timestamp'].dt.date <= end_date)
    ]
    
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Analyses", len(filtered_df))
    col2.metric("Total Packets", filtered_df['Total Packets'].sum())
    col3.metric("Total Intrusions", filtered_df['Intrusions'].sum())
    
    st.subheader("Intrusion Trends")
    fig = px.line(
        filtered_df,
        x='Timestamp',
        y=['Intrusions', 'Normal'],
        title="Intrusion and Normal Traffic Over Time",
        labels={'value': 'Count', 'variable': 'Traffic Type'}
    )
    fig.update_layout(
        paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
        plot_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
        font_color='#ffffff'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Attack Type Distribution")
    all_results = []
    for h in st.session_state.analysis_history:
        if h['timestamp'].date() >= start_date and h['timestamp'].date() <= end_date:
            all_results.extend(h['results'])
    
    if all_results:
        attack_types = pd.Series([r['prediction'] for r in all_results if r['is_intrusion']]).value_counts()
        fig = px.pie(
            names=attack_types.index,
            values=attack_types.values,
            title="Distribution of Attack Types"
        )
        fig.update_layout(
            paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
            font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_alert_log():
    st.header("Alert Log")
    
    if not st.session_state.alert_log:
        st.info("No alerts generated yet. Run a simulation with a high confidence threshold.")
        return
    
    st.markdown("""
    View intrusion alerts with timestamps and details.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Alerts are triggered when intrusions exceed the confidence threshold.</span>
    </div>
    """, unsafe_allow_html=True)
    
    alert_df = pd.DataFrame(st.session_state.alert_log)
    st.dataframe(alert_df[['timestamp', 'message', 'recipient']], use_container_width=True)
    
    if st.button("Clear Alert Log"):
        st.session_state.alert_log = []
        st.success("Alert log cleared.")

def show_retrain_model():
    global model, scaler, label_encoders, le_class, model_type
    
    st.header("Retrain Model")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    Improve the model using feedback from analyses.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Active learning prioritizes uncertain predictions for feedback.</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.feedback_data:
        st.info("No feedback data available. Provide feedback on analysis results first.")
        return
    
    feedback_df = pd.DataFrame(st.session_state.feedback_data)
    st.dataframe(feedback_df[[col for col in nsl_kdd_columns if col in feedback_df.columns]], use_container_width=True)
    
    st.subheader("Retrain Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of estimators", 50, 500, 200, 50)
        max_depth = st.slider("Max depth", 3, 10, 6)
    with col2:
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        scale_pos_weight = st.slider("Scale positive weight", 1, 5, 2)
    
    if st.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            try:
                train_file = st.session_state.get('last_train_file')
                if train_file:
                    data = pd.read_csv(train_file, names=nsl_kdd_columns, header=None)
                else:
                    data = pd.DataFrame(columns=nsl_kdd_columns)
                
                data = pd.concat([data, feedback_df], ignore_index=True)
                
                data, label_encoders, le_class = preprocess_data(data, {}, None, is_train=True)
                
                X = data.drop('class', axis=1)
                y = data['class']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                
                if model_type == 'LSTM':
                    model = train_lstm_model(X_train, y_train, X_test, y_test)
                else:
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                
                if model_type == 'LSTM':
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = (model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int).flatten()
                else:
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                target_names = [le_class.classes_[i] for i in unique_labels]
                report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                
                joblib.dump(model, 'idps_model.pkl')
                joblib.dump(scaler, 'scaler.pkl')
                joblib.dump(label_encoders, 'label_encoders.pkl')
                joblib.dump(le_class, 'le_class.pkl')
                joblib.dump(model_type, 'model_type.pkl')
                
                st.success("Model retrained successfully!")
                st.metric("Accuracy", f"{accuracy:.2%}")
                st.text(report)
                
            except Exception as e:
                st.error(f"Error during retraining: {str(e)}")

def show_train_model():
    global model, scaler, label_encoders, le_class, autoencoder, model_type
    
    st.header("Train New Model")
    
    if model is not None:
        st.warning("A trained model exists. Training a new model will overwrite it.")
    
    st.markdown("""
    Train a new IDPS model using the NSL-KDD dataset.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Upload training data and select model type to start.</span>
    </div>
    """, unsafe_allow_html=True)
    
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    
    model_type = st.selectbox("Model Type", ["XGBoost", "LSTM"], index=0)
    
    st.subheader("Model Parameters")
    if model_type == "XGBoost":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of estimators", 50, 500, 200, 50)
            max_depth = st.slider("Max depth", 3, 10, 6)
        with col2:
            learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
            scale_pos_weight = st.slider("Scale positive weight", 1, 5, 2)
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42
        }
    else:
        params = {}
    
    train_autoencoder_option = st.checkbox("Train Autoencoder for Anomaly Detection")
    
    if st.button("Train Model"):
        if train_file is not None:
            with st.spinner("Training model..."):
                try:
                    data = pd.read_csv(train_file, names=nsl_kdd_columns, header=None)
                    data, label_encoders, le_class = preprocess_data(data, {}, None, is_train=True)
                    
                    X = data.drop('class', axis=1)
                    y = data['class']
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    
                    if model_type == "LSTM":
                        model = train_lstm_model(X_train, y_train, X_test, y_test)
                        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        y_pred = (model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int).flatten()
                        y_pred_prob = model.predict(X_test_reshaped, verbose=0)[:, 0]
                    else:
                        model = XGBClassifier(**params)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_prob = model.predict_proba(X_test)[:, 1]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                    target_names = [le_class.classes_[i] for i in unique_labels]
                    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                    
                    if train_autoencoder_option:
                        X_normal = X_train[y_train == le_class.transform(['normal'])[0]]
                        autoencoder = train_autoencoder(scaler.transform(X_normal))
                        joblib.dump(autoencoder, 'autoencoder.pkl')
                    
                    joblib.dump(model, 'idps_model.pkl')
                    joblib.dump(scaler, 'scaler.pkl')
                    joblib.dump(label_encoders, 'label_encoders.pkl')
                    joblib.dump(le_class, 'le_class.pkl')
                    joblib.dump(model_type, 'model_type.pkl')
                    
                    st.session_state['last_train_file'] = train_file
                    
                    st.success(f"{model_type} model trained successfully!")
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Model Performance")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=le_class.transform(['normal'])[0])
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.line(
                        x=fpr, y=tpr,
                        title=f"ROC Curve (AUC = {roc_auc:.2f})",
                        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                    )
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    fig_roc.update_layout(
                        paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        plot_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        font_color='#ffffff'
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob, pos_label=le_class.transform(['normal'])[0])
                    fig_pr = px.line(
                        x=recall, y=precision,
                        title="Precision-Recall Curve",
                        labels={'x': 'Recall', 'y': 'Precision'}
                    )
                    fig_pr.update_layout(
                        paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        plot_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        font_color='#ffffff'
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
                    
                    st.text(report)
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please upload a training data file.")

def show_test_model():
    global model, scaler, label_encoders, le_class, model_type
    
    st.header("Test Model")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    Test the IDPS model with a test dataset or manual input.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Upload test data or enter traffic parameters to evaluate model performance.</span>
    </div>
    """, unsafe_allow_html=True)
    
    test_option = st.radio("Select testing method", ["Upload Test File", "Manual Input"])
    
    if test_option == "Upload Test File":
        test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
        
        if test_file is not None and st.button("Run Test"):
            with st.spinner("Testing model..."):
                try:
                    test_data = pd.read_csv(test_file, names=nsl_kdd_columns, header=None)
                    test_data, _, _ = preprocess_data(test_data, label_encoders, le_class, is_train=False)
                    
                    X_test = test_data.drop('class', axis=1)
                    y_test = test_data['class']
                    
                    X_test_scaled = scaler.transform(X_test)
                    
                    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
                    if model_type == 'LSTM':
                        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
                        y_pred_prob = model.predict(X_test_reshaped, verbose=0)[:, 0]
                        y_pred = (y_pred_prob >= threshold).astype(int).flatten()
                    else:
                        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
                        y_pred = (y_pred_prob >= threshold).astype(int)
                    
                    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                    target_names = [le_class.classes_[i] for i in unique_labels]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                    
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Classification Report")
                    st.text(report)
                    
                    st.subheader("Confusion Matrix")
                    conf_matrix = pd.crosstab(
                        le_class.inverse_transform(y_test),
                        le_class.inverse_transform(y_pred),
                        rownames=['Actual'],
                        colnames=['Predicted']
                    )
                    fig_cm = px.imshow(
                        conf_matrix,
                        text_auto=True,
                        title="Confusion Matrix",
                        color_continuous_scale='Blues'
                    )
                    fig_cm.update_layout(
                        paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        plot_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                        font_color='#ffffff'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during testing: {str(e)}")
    
    else:
        st.subheader("Enter Network Traffic Parameters")
        
        with st.form("manual_test_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                duration = st.number_input("Duration", min_value=0, value=0)
                protocol_type = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
                service = st.selectbox("Service", ['http', 'smtp', 'ftp', 'other'])
                flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'other'])
                src_bytes = st.number_input("Source Bytes", min_value=0, value=100)
                dst_bytes = st.number_input("Destination Bytes", min_value=0, value=200)
                wrong_fragment = st.number_input("Wrong Fragment", min_value=0, value=0)
                hot = st.number_input("Hot", min_value=0, value=0)
                
            with col2:
                logged_in = st.selectbox("Logged In", [0, 1], format_func=lambda x: "Yes" if x else "No")
                num_compromised = st.number_input("Num Compromised", min_value=0, value=0)
                count = st.number_input("Count", min_value=0, value=2)
                srv_count = st.number_input("Service Count", min_value=0, value=2)
                serror_rate = st.number_input("Serror Rate", min_value=0.0, max_value=1.0, value=0.0)
                srv_serror_rate = st.number_input("Service Serror Rate", min_value=0.0, max_value=1.0, value=0.0)
                rerror_rate = st.number_input("Rerror Rate", min_value=0.0, max_value=1.0, value=0.0)
                srv_rerror_rate = st.number_input("Service Rerror Rate", min_value=0.0, max_value=1.0, value=0.0)
            
            threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
            
            if st.form_submit_button("Test Traffic"):
                input_data = {
                    'duration': [duration],
                    'protocol_type': [protocol_type],
                    'service': [service],
                    'flag': [flag],
                    'src_bytes': [src_bytes],
                    'dst_bytes': [dst_bytes],
                    'wrong_fragment': [wrong_fragment],
                    'hot': [hot],
                    'logged_in': [logged_in],
                    'num_compromised': [num_compromised],
                    'count': [count],
                    'srv_count': [srv_count],
                    'serror_rate': [serror_rate],
                    'srv_serror_rate': [srv_serror_rate],
                    'rerror_rate': [rerror_rate],
                    'srv_rerror_rate': [srv_rerror_rate],
                    'same_srv_rate': [1.0],
                    'diff_srv_rate': [0.0],
                    'srv_diff_host_rate': [0.0],
                    'dst_host_count': [count],
                    'dst_host_srv_count': [srv_count],
                    'dst_host_same_srv_rate': [1.0],
                    'dst_host_diff_srv_rate': [0.0],
                    'dst_host_same_src_port_rate': [0.0],
                    'dst_host_srv_diff_host_rate': [0.0],
                    'dst_host_serror_rate': [serror_rate],
                    'dst_host_srv_serror_rate': [srv_serror_rate],
                    'dst_host_rerror_rate': [rerror_rate],
                    'dst_host_srv_rerror_rate': [srv_rerror_rate]
                }
                
                input_df = pd.DataFrame(input_data)
                
                prediction, confidence = predict_traffic(input_df, threshold)
                
                if prediction is not None and confidence is not None:
                    if prediction == 'normal':
                        st.success(f"Normal traffic detected (Confidence: {confidence:.2%})")
                    else:
                        st.error(f"Intrusion detected: {prediction} (Confidence: {confidence:.2%})")
                    
                    explanation = explain_threat(prediction, confidence, "Manual Input", st.session_state.xai_api_key, st.session_state.openai_api_key)
                    st.markdown(f"**Explanation**: {explanation}")

def show_realtime_detection():
    global model, scaler, label_encoders, le_class, model_type
    
    st.header("Real-time Intrusion Detection Simulation")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    Simulate real-time network traffic monitoring.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Generate random traffic to test the model's real-time detection capabilities.</span>
    </div>
    """, unsafe_allow_html=True)
    
    simulation_speed = st.select_slider("Simulation Speed", options=["Slow", "Normal", "Fast"], value="Normal")
    num_samples = st.slider("Number of samples to simulate", 10, 100, 20)
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
    
    if st.button("Start Simulation"):
        np.random.seed(42)
        sample_data = {
            'duration': np.random.randint(0, 100, num_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], num_samples),
            'service': np.random.choice(['http', 'smtp', 'ftp', 'other'], num_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'other'], num_samples),
            'src_bytes': np.random.randint(0, 10000, num_samples),
            'dst_bytes': np.random.randint(0, 10000, num_samples),
            'wrong_fragment': np.random.randint(0, 5, num_samples),
            'hot': np.random.randint(0, 5, num_samples),
            'logged_in': np.random.choice([0, 1], num_samples),
            'num_compromised': np.random.randint(0, 5, num_samples),
            'count': np.random.randint(0, 10, num_samples),
            'srv_count': np.random.randint(0, 10, num_samples),
            'serror_rate': np.random.uniform(0, 1, num_samples),
            'srv_serror_rate': np.random.uniform(0, 1, num_samples),
            'rerror_rate': np.random.uniform(0, 1, num_samples),
            'srv_rerror_rate': np.random.uniform(0, 1, num_samples),
            'same_srv_rate': np.random.uniform(0, 1, num_samples),
            'diff_srv_rate': np.random.uniform(0, 1, num_samples),
            'srv_diff_host_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_count': np.random.randint(0, 10, num_samples),
            'dst_host_srv_count': np.random.randint(0, 10, num_samples),
            'dst_host_same_srv_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_diff_srv_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_same_src_port_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srv_diff_host_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_serror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srv_serror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_rerror_rate': np.random.uniform(0, 1, num_samples),
            'dst_host_srv_rerror_rate': np.random.uniform(0, 1, num_samples)
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        
        results = []
        intrusion_count = 0
        
        delay = 1.0 if simulation_speed == "Slow" else 0.5 if simulation_speed == "Normal" else 0.1
        
        for i in range(num_samples):
            current_sample = sample_df.iloc[[i]]
            
            prediction, confidence = predict_traffic(current_sample, threshold)
            
            if prediction is None or confidence is None:
                continue
            
            is_intrusion = prediction != 'normal'
            if is_intrusion:
                intrusion_count += 1
                st.balloons()
                st.markdown(f"<div style='background-color:#ff4d4d;padding:10px;border-radius:5px;color:#ffffff;'>🚨 Intrusion detected: {prediction} (Confidence: {confidence:.2%})</div>", unsafe_allow_html=True)
            results.append(is_intrusion)
            
            status_placeholder.markdown(f"""
            **Processing sample {i+1}/{num_samples}**  
            Last detection: {'Intrusion' if is_intrusion else 'Normal'}  
            Total intrusions detected: {intrusion_count}
            """)
            
            fig = px.line(
                x=range(1, len(results) + 1),
                y=results,
                title="Detection Results Over Time",
                labels={'x': 'Sample', 'y': 'Status'},
            )
            fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=['Normal', 'Intrusion'])
            fig.update_layout(
                paper_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                plot_bgcolor='#2a2a3d' if st.session_state.theme == 'Dark' else '#ffffff',
                font_color='#ffffff'
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            log_placeholder.markdown(f"**Sample {i+1}**: {'🚨 Intrusion' if is_intrusion else '✅ Normal'} (Confidence: {confidence:.2%})")
            
            time.sleep(delay)
        
        st.success(f"Simulation complete! Detected {intrusion_count} intrusions out of {num_samples} samples.")

def show_threat_intelligence():
    st.header("Threat Intelligence Feed")
    
    st.markdown("""
    Stay updated with the latest cybersecurity threats.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Mock feed showcasing recent threats for demonstration purposes.</span>
    </div>
    """, unsafe_allow_html=True)
    
    mock_feed = [
        {"date": "2025-05-01", "threat": "DDoS Attack", "description": "Increased botnet activity targeting cloud services.", "severity": "High"},
        {"date": "2025-04-30", "threat": "Phishing Campaign", "description": "Spear-phishing emails targeting corporate users.", "severity": "Medium"},
        {"date": "2025-04-29", "threat": "Ransomware", "description": "New variant exploiting unpatched systems.", "severity": "Critical"}
    ]
    
    for item in mock_feed:
        st.markdown(f"""
        **{item['date']} - {item['threat']}** ({item['severity']})  
        {item['description']}
        """)

def show_documentation():
    st.header("Project Documentation")
    
    st.markdown("""
    ### AI-Enhanced Intrusion Detection and Prevention System
    
    **Overview**  
    This IDPS leverages machine learning (XGBoost, LSTM, Autoencoders) and Generative AI to detect and prevent network intrusions. Built for academic and professional use, it offers a user-friendly interface and advanced analytics.
    
    **Objectives**  
    - Detect network intrusions with high accuracy using ML models.
    - Provide actionable insights with Generative AI explanations.
    - Enable real-time monitoring and port scanning capabilities.
    
    **System Architecture**  
    - **Data Ingestion**: Processes NSL-KDD datasets and simulates NMAP scans.
    - **ML Models**: XGBoost for classification, LSTM for temporal patterns, Autoencoders for anomaly detection.
    - **Generative AI**: Integrates xAI/OpenAI APIs for threat explanations.
    - **Frontend**: Streamlit with Plotly for interactive visualizations.
    - **Storage**: In-memory session state for scalability.
    
    **Key Features**  
    - Real-time intrusion detection simulation.
    - NMAP-like port scanning interface.
    - Interactive dashboards with Plotly visualizations.
    - Multi-format report export (CSV).
    - Active learning for model improvement.
    - Mock threat intelligence feed.
    
    **Methodology**  
    - **Dataset**: NSL-KDD with 41 features and attack labels (DoS, Probe, R2L, U2R).
    - **Preprocessing**: Label encoding, SMOTE for class imbalance, feature scaling.
    - **Evaluation**: Accuracy, ROC-AUC, Precision-Recall curves.
    
    **Technology Stack**  
    - Python, Streamlit, Scikit-learn, TensorFlow, XGBoost, Plotly, ReportLab.
    
    **Future Improvements**  
    - Integrate live NMAP scanning (requires network permissions).
    - Support additional datasets (e.g., CICIDS2017).
    - Enhance Generative AI with custom prompts.
    
    **Contact**  
    For feedback, contact the developer via [email@example.com](mailto:email@example.com).
    """)

def main():
    st.sidebar.image("https://via.placeholder.com/150?text=IDPS+Logo", use_container_width=True)
    st.sidebar.title("AI-Enhanced IDPS")
    st.sidebar.button("Toggle Theme", on_click=toggle_theme)
    
    app_mode = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Train Model", "Test Model", "Real-time Detection", "NMAP Analysis", "Historical Analysis", "Alert Log", "Retrain Model", "Threat Intelligence", "Documentation"],
        format_func=lambda x: f"{'🛠️' if x in ['Train Model', 'Retrain Model'] else '📊' if x in ['Historical Analysis', 'Test Model'] else '🚨' if x == 'Alert Log' else '🌐' if x == 'Real-time Detection' else '🔍' if x == 'NMAP Analysis' else 'ℹ️' if x == 'Documentation' else '📰' if x == 'Threat Intelligence' else '🏠'} {x}"
    )
    
    if app_mode == "Home":
        st.header("AI-Enhanced Intrusion Detection and Prevention System")
        st.markdown("""
        Welcome to a state-of-the-art IDPS powered by AI. Detect and prevent network intrusions with advanced machine learning and Generative AI.
        
        ### Features
        - **Deep Learning**: LSTM for temporal pattern detection.
        - **Anomaly Detection**: Autoencoders for zero-day attacks.
        - **Generative AI**: Threat explanations via xAI/OpenAI APIs.
        - **Interactive Dashboards**: Visualize threats with Plotly.
        - **NMAP Analysis**: Simulate port scanning to identify open services.
        - **Real-time Alerts**: Instant notifications for intrusions.
        
        Start by training a model or running an NMAP scan!
        """)
        if model is not None:
            st.success(f"Loaded {model_type} model is ready!")
        else:
            st.warning("Train a model to begin.")
    
    elif app_mode == "Train Model":
        show_train_model()
    elif app_mode == "Test Model":
        show_test_model()
    elif app_mode == "Real-time Detection":
        show_realtime_detection()
    elif app_mode == "NMAP Analysis":
        show_nmap_analysis()
    elif app_mode == "Historical Analysis":
        show_historical_analysis()
    elif app_mode == "Alert Log":
        show_alert_log()
    elif app_mode == "Retrain Model":
        show_retrain_model()
    elif app_mode == "Threat Intelligence":
        show_threat_intelligence()
    elif app_mode == "Documentation":
        show_documentation()

if __name__ == "__main__":
    main()
