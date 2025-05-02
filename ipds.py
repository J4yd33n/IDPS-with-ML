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

# Try importing Scapy
try:
    from scapy.all import rdpcap, TCP, UDP, ICMP, IP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Set page config
st.set_page_config(page_title="AI-Enhanced IDPS", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for professional look
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e2f;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>input {
        border-radius: 5px;
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
        color: #fff;
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
</style>
""", unsafe_allow_html=True)

# NSL-KDD columns
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

# Theme toggle
def toggle_theme():
    st.session_state.theme = "Dark" if st.session_state.theme == "Light" else "Light"
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {'#1e1e2f' if st.session_state.theme == 'Dark' else '#f0f2f6'};
                color: {'#ffffff' if st.session_state.theme == 'Dark' else '#000000'};
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

def process_packet(packet, connection_tracker, time_window=10):
    features = {col: 0 for col in nsl_kdd_columns if col != 'class'}
    
    if IP not in packet:
        return None
    
    features['protocol_type'] = packet[IP].proto
    if packet[IP].proto == 6:
        features['protocol_type'] = 'tcp'
        if TCP in packet:
            features['src_bytes'] = len(packet[TCP].payload)
            features['dst_bytes'] = 0
            features['service'] = str(packet[TCP].dport)
            features['flag'] = 'SF'
    elif packet[IP].proto == 17:
        features['protocol_type'] = 'udp'
        if UDP in packet:
            features['src_bytes'] = len(packet[UDP].payload)
            features['dst_bytes'] = 0
            features['service'] = str(packet[UDP].dport)
            features['flag'] = 'SF'
    elif packet[IP].proto == 1:
        features['protocol_type'] = 'icmp'
        features['src_bytes'] = len(packet[ICMP])
        features['dst_bytes'] = 0
        features['service'] = 'other'
        features['flag'] = 'SF'
    
    service_map = {
        '80': 'http', '443': 'http', '21': 'ftp', '22': 'ssh', '23': 'telnet',
        '25': 'smtp', '53': 'dns', '110': 'pop3', '143': 'imap'
    }
    features['service'] = service_map.get(features['service'], 'other')
    
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    dst_port = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
    timestamp = packet.time
    
    connection_key = (src_ip, dst_ip, dst_port, features['protocol_type'])
    current_time = datetime.fromtimestamp(timestamp)
    connection_tracker[connection_key] = [
        pkt for pkt in connection_tracker.get(connection_key, [])
        if datetime.fromtimestamp(pkt['time']) > current_time - timedelta(seconds=time_window)
    ]
    connection_tracker[connection_key].append({'time': timestamp, 'packet': packet})
    
    features['count'] = len(connection_tracker[connection_key])
    features['srv_count'] = len([
        pkt for pkt in connection_tracker.get((dst_ip, src_ip, dst_port, features['protocol_type']), [])
    ])
    features['same_srv_rate'] = features['count'] / max(1, features['srv_count'])
    features['diff_srv_rate'] = 1 - features['same_srv_rate']
    
    features['dst_host_count'] = len(set(
        pkt['packet'][IP].src for pkt in connection_tracker.get((dst_ip, src_ip, dst_port, features['protocol_type']), [])
    ))
    features['dst_host_srv_count'] = features['srv_count']
    features['dst_host_same_srv_rate'] = features['same_srv_rate']
    
    return features

def generate_pdf_report(analysis_results, intrusion_count, total_packets, filename, temp_dir="temp"):
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, f"IDPS_Report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("AI-Enhanced IDPS Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    summary_text = f"""
    <b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>PCAP File:</b> {filename}<br/>
    <b>Total Packets Analyzed:</b> {total_packets}<br/>
    <b>Intrusions Detected:</b> {intrusion_count}<br/>
    <b>Normal Traffic:</b> {total_packets - intrusion_count}
    """
    elements.append(Paragraph(summary_text, styles['BodyText']))
    elements.append(Spacer(1, 12))
    
    timeline_path = os.path.join(temp_dir, "intrusion_timeline.png")
    fig = px.line(
        x=range(1, len(analysis_results) + 1),
        y=[1 if r['is_intrusion'] else 0 for r in analysis_results],
        labels={'x': 'Packet Number', 'y': 'Status'},
        title="Intrusion Detection Timeline"
    )
    fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 1], ticktext=['Normal', 'Intrusion'])
    fig.write_to_file(timeline_path)
    elements.append(ReportLabImage(timeline_path, width=400, height=200))
    elements.append(Spacer(1, 12))
    
    src_ip_counts = pd.Series([r['src_ip'] for r in analysis_results if r['is_intrusion']]).value_counts().head(5)
    if not src_ip_counts.empty:
        ip_plot_path = os.path.join(temp_dir, "src_ip_distribution.png")
        fig = px.bar(
            x=src_ip_counts.index,
            y=src_ip_counts.values,
            labels={'x': 'Source IP', 'y': 'Intrusion Count'},
            title="Top 5 Source IPs with Intrusions"
        )
        fig.write_to_file(ip_plot_path)
        elements.append(ReportLabImage(ip_plot_path, width=400, height=200))
        elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Detailed Results", styles['Heading2']))
    data = [['Packet', 'Source IP', 'Prediction', 'Confidence', 'Anomaly', 'Explanation']]
    for r in analysis_results[:10]:
        data.append([
            str(r['packet_num']),
            r['src_ip'],
            r['prediction'],
            f"{r['confidence']:.2%}",
            'Yes' if r['is_anomaly'] else 'No',
            r['explanation'][:50] + '...' if len(r['explanation']) > 50 else r['explanation']
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
    elements.append(table)
    
    elements.append(Paragraph("Recommendations", styles['Heading2']))
    rec_text = "Based on the analysis:<br/>"
    if intrusion_count > 0:
        rec_text += f"- Block the top source IPs: {', '.join(src_ip_counts.index)}.<br/>"
        rec_text += "- Review firewall rules and enable deep packet inspection.<br/>"
    else:
        rec_text += "- No intrusions detected; maintain regular monitoring."
    elements.append(Paragraph(rec_text, styles['BodyText']))
    
    doc.build(elements)
    return pdf_path

def generate_csv_report(analysis_results):
    df = pd.DataFrame(analysis_results)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def generate_json_report(analysis_results):
    df = pd.DataFrame(analysis_results)
    return df.to_json(orient='records')

def select_uncertain_packets(analysis_results, uncertainty_threshold=0.1):
    return [i for i, r in enumerate(analysis_results) if abs(r['confidence'] - 0.5) < uncertainty_threshold]

def get_sample_pcap():
    sample_data = """
    0000   45 00 00 34 00 01 00 00 40 06 3a 8c c0 a8 01 02
    0010   c0 a8 01 03 00 50 00 15 00 00 00 00 50 10 20 00
    0020   8e 8f 00 00 47 45 54 20 2f 20 48 54 54 50 2f 31
    0030   2e 31 0d 0a 0d 0a
    """
    temp_pcap = "sample.pcap"
    with open(temp_pcap, "wb") as f:
        f.write(bytes.fromhex(''.join(sample_data.split())))
    return temp_pcap

def show_pcap_analysis():
    global model, scaler, label_encoders, le_class, autoencoder, model_type
    
    st.header("PCAP File Analysis")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    if not SCAPY_AVAILABLE:
        st.error("Scapy is not installed. Please install it with 'pip install scapy'.")
        return
    
    st.markdown("""
    Analyze network traffic from PCAP files for intrusions using AI models (LSTM, Autoencoders, Active Learning).
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Upload a PCAP file or use the sample to detect intrusions. API keys enhance explanations.</span>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["Upload PCAP", "Use Sample PCAP"])
    
    with tabs[0]:
        pcap_file = st.file_uploader("Upload PCAP File", type=["pcap", "pcapng"])
    
    with tabs[1]:
        if st.button("Use Sample PCAP"):
            pcap_file = get_sample_pcap()
            st.success("Sample PCAP loaded for analysis.")
    
    st.subheader("API Keys")
    col1, col2 = st.columns(2)
    with col1:
        xai_api_key = st.text_input("xAI API Key", type="password", value=st.session_state.xai_api_key)
        if xai_api_key:
            st.session_state.xai_api_key = xai_api_key
    with col2:
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
    
    st.subheader("Settings")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
    alert_threshold = st.slider("Alert Confidence Threshold", 0.5, 0.9, 0.7, 0.05)
    
    if (pcap_file is not None or os.path.exists("sample.pcap")) and st.button("Analyze PCAP"):
        with st.spinner("Analyzing PCAP file..."):
            progress_bar = st.progress(0)
            try:
                if isinstance(pcap_file, str):
                    packets = rdpcap(pcap_file)
                    filename = "sample.pcap"
                else:
                    temp_pcap = "temp_upload.pcap"
                    with open(temp_pcap, "wb") as f:
                        f.write(pcap_file.read())
                    packets = rdpcap(temp_pcap)
                    filename = pcap_file.name
                
                analysis_results = []
                intrusion_count = 0
                
                for i, packet in enumerate(packets):
                    features = process_packet(packet, st.session_state.connection_tracker)
                    if features is None:
                        continue
                    
                    input_df = pd.DataFrame([features])
                    prediction, confidence = predict_traffic(input_df, threshold)
                    
                    if prediction is None or confidence is None:
                        continue
                    
                    is_intrusion = prediction != 'normal'
                    src_ip = packet[IP].src if IP in packet else "Unknown"
                    is_anomaly = False
                    if autoencoder:
                        input_data_scaled = scaler.transform(input_df)
                        is_anomaly = detect_anomaly(input_data_scaled, autoencoder)[0]
                    
                    explanation = explain_threat(prediction, confidence, src_ip, st.session_state.xai_api_key, st.session_state.openai_api_key)
                    
                    if is_intrusion and confidence >= alert_threshold:
                        intrusion_count += 1
                        alert_message = f"Intrusion detected in packet {i+1} ({prediction}) from {src_ip} with confidence {confidence:.2%}"
                        st.session_state.alert_log.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'message': alert_message,
                            'recipient': 'admin@example.com'
                        })
                        st.balloons()
                        st.markdown(f"<div style='background-color:#ff4d4d;padding:10px;border-radius:5px;'>🚨 {alert_message}</div>", unsafe_allow_html=True)
                    
                    analysis_results.append({
                        'packet_num': i+1,
                        'is_intrusion': is_intrusion,
                        'prediction': prediction,
                        'confidence': confidence,
                        'src_ip': src_ip,
                        'is_anomaly': is_anomaly,
                        'explanation': explanation
                    })
                    progress_bar.progress((i + 1) / len(packets))
                
                history_entry = {
                    'timestamp': datetime.now(),
                    'filename': filename,
                    'total_packets': len(packets),
                    'intrusion_count': intrusion_count,
                    'results': analysis_results
                }
                st.session_state.analysis_history.append(history_entry)
                
                st.success(f"Analysis complete: {intrusion_count} intrusions detected in {len(packets)} packets.")
                
                st.subheader("Interactive Dashboard")
                result_df = pd.DataFrame(analysis_results)
                
                fig_heatmap = px.density_heatmap(
                    result_df[result_df['is_intrusion']],
                    x='confidence',
                    y='prediction',
                    title="Intrusion Confidence Heatmap",
                    labels={'confidence': 'Confidence', 'prediction': 'Attack Type'}
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                fig_treemap = px.treemap(
                    result_df[result_df['is_intrusion']],
                    path=['prediction', 'src_ip'],
                    values='confidence',
                    title="Attack Type and Source IP Treemap"
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
                
                st.subheader("Results Table")
                st.dataframe(result_df[['packet_num', 'src_ip', 'prediction', 'confidence', 'is_anomaly', 'explanation']], use_container_width=True)
                
                st.subheader("Export Report")
                col_pdf, col_csv, col_json = st.columns(3)
                with col_pdf:
                    pdf_path = generate_pdf_report(analysis_results, intrusion_count, len(packets), filename)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download PDF",
                            data=f,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                with col_csv:
                    csv_data = generate_csv_report(analysis_results)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"IDPS_Report_{filename}.csv",
                        mime="text/csv"
                    )
                with col_json:
                    json_data = generate_json_report(analysis_results)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"IDPS_Report_{filename}.json",
                        mime="application/json"
                    )
                
                st.subheader("Feedback")
                with st.form(f"feedback_form_{filename}"):
                    uncertain_indices = select_uncertain_packets(analysis_results)
                    feedback_indices = st.multiselect(
                        "Select packets to mark as incorrect",
                        options=result_df.index,
                        default=uncertain_indices,
                        format_func=lambda x: f"Packet {result_df.loc[x, 'packet_num']} ({result_df.loc[x, 'prediction']}, Confidence: {result_df.loc[x, 'confidence']:.2%})"
                    )
                    correct_label = st.selectbox("Correct Label", ['normal'] + list(le_class.classes_))
                    if st.form_submit_button("Submit Feedback"):
                        for idx in feedback_indices:
                            packet_features = process_packet(packets[result_df.loc[idx, 'packet_num'] - 1], st.session_state.connection_tracker)
                            if packet_features:
                                packet_features['class'] = correct_label
                                st.session_state.feedback_data.append(packet_features)
                        st.success(f"Feedback saved for {len(feedback_indices)} packets.")
                
                if isinstance(pcap_file, str):
                    os.remove(pcap_file)
                else:
                    os.remove(temp_pcap)
                
            except Exception as e:
                st.error(f"Error during PCAP analysis: {str(e)}")
                if os.path.exists("temp_upload.pcap"):
                    os.remove("temp_upload.pcap")
                if os.path.exists("sample.pcap"):
                    os.remove("sample.pcap")

def show_historical_analysis():
    st.header("Historical Analysis Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Analyze a PCAP file first.")
        return
    
    st.markdown("""
    Visualize trends and insights from past PCAP analyses.
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
        st.plotly_chart(fig, use_container_width=True)

def show_alert_log():
    st.header("Alert Log")
    
    if not st.session_state.alert_log:
        st.info("No alerts generated yet. Analyze a PCAP file with a high confidence threshold.")
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
    Improve the model using feedback from PCAP analyses.
    <div class="tooltip">ℹ️
        <span class="tooltiptext">Active learning prioritizes uncertain predictions for feedback.</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.feedback_data:
        st.info("No feedback data available. Provide feedback on PCAP analysis results first.")
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
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob, pos_label=le_class.transform(['normal'])[0])
                    fig_pr = px.line(
                        x=recall, y=precision,
                        title="Precision-Recall Curve",
                        labels={'x': 'Recall', 'y': 'Precision'}
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
                st.markdown(f"<div style='background-color:#ff4d4d;padding:10px;border-radius:5px;'>🚨 Intrusion detected: {prediction} (Confidence: {confidence:.2%})</div>", unsafe_allow_html=True)
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
    - Enable real-time monitoring and historical analysis.
    
    **System Architecture**  
    - **Data Ingestion**: Processes NSL-KDD datasets and PCAP files using Scapy.
    - **ML Models**: XGBoost for classification, LSTM for temporal patterns, Autoencoders for anomaly detection.
    - **Generative AI**: Integrates xAI/OpenAI APIs for threat explanations.
    - **Frontend**: Streamlit with Plotly for interactive visualizations.
    - **Storage**: In-memory session state for scalability.
    
    **Key Features**  
    - Real-time intrusion detection simulation.
    - Interactive dashboards with Plotly visualizations.
    - Multi-format report export (PDF, CSV, JSON).
    - Active learning for model improvement.
    - Mock threat intelligence feed.
    
    **Methodology**  
    - **Dataset**: NSL-KDD with 41 features and attack labels (DoS, Probe, R2L, U2R).
    - **Preprocessing**: Label encoding, SMOTE for class imbalance, feature scaling.
    - **Evaluation**: Accuracy, ROC-AUC, Precision-Recall curves.
    
    **Technology Stack**  
    - Python, Streamlit, Scikit-learn, TensorFlow, XGBoost, Scapy, Plotly, ReportLab.
    
    **Future Improvements**  
    - Integrate live network capture with Scapy.
    - Support additional datasets (e.g., CICIDS2017).
    - Enhance Generative AI with custom prompts.
    
    **Contact**  
    For feedback, contact the developer via [email@example.com](mailto:email@example.com).
    """)

def main():
    st.sidebar.image("https://via.placeholder.com/150?text=IDPS+Logo", use_column_width=True)
    st.sidebar.title("AI-Enhanced IDPS")
    st.sidebar.button("Toggle Theme", on_click=toggle_theme)
    
    app_mode = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Train Model", "Test Model", "Real-time Detection", "PCAP Analysis", "Historical Analysis", "Alert Log", "Retrain Model", "Threat Intelligence", "Documentation"],
        format_func=lambda x: f"{'🛠️' if x in ['Train Model', 'Retrain Model'] else '📊' if x in ['Historical Analysis', 'Test Model'] else '🚨' if x == 'Alert Log' else '🌐' if x == 'Real-time Detection' else '🔍' if x == 'PCAP Analysis' else 'ℹ️' if x == 'Documentation' else '📰' if x == 'Threat Intelligence' else '🏠'} {x}"
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
        - **Real-time Alerts**: Instant notifications for intrusions.
        
        Start by training a model or analyzing a PCAP file!
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
    elif app_mode == "PCAP Analysis":
        show_pcap_analysis()
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
