import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time
import os
from datetime import datetime, timedelta
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
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
st.set_page_config(page_title="AI-Enhanced IDPS", page_icon="🛡️", layout="wide")

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
        st.write("Model and preprocessing objects loaded successfully.")
        return model, scaler, label_encoders, le_class, autoencoder, model_type
    except Exception as e:
        st.error(f"Failed to load model or preprocessing objects: {str(e)}")
        return None, None, None, None, None, 'XGBoost'

model, scaler, label_encoders, le_class, autoencoder, model_type = load_model()

def preprocess_data(df, label_encoders, le_class, is_train=True):
    df = df.copy()
    
    # Handle missing values
    df.fillna({'protocol_type': 'missing', 'service': 'missing', 'flag': 'missing'}, inplace=True)
    df.fillna(0, inplace=True)
    
    # Convert numeric columns
    numeric_cols = [
        col for col in nsl_kdd_columns 
        if col not in categorical_cols + ['class'] + low_importance_features
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Encode categorical features
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
    
    # Encode target if exists
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
    
    # Drop low importance features
    df = df.drop(columns=[col for col in low_importance_features if col in df.columns], errors='ignore')
    
    return df, label_encoders,atação

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
    model.fit(X_train_reshaped, y_train, epochs=5, batch_size=16, validation_data=(X_test_reshaped, y_test), verbose=0)
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
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=16, verbose=0)
    return autoencoder

def detect_anomaly(input_data_scaled, autoencoder, threshold=0.1):
    reconstructions = autoencoder.predict(input_data_scaled)
    mse = np.mean(np.square(input_data_scaled - reconstructions), axis=1)
    return mse > threshold

def predict_traffic(input_data, threshold=0.5):
    global model, scaler, label_encoders, le_class, model_type
    
    if model is None or scaler is None or label_encoders is None or le_class is None:
        st.error("Model or preprocessing components not loaded.")
        return None, None
    
    try:
        input_data = input_data.copy()
        
        # Preprocess categorical features
        for col in categorical_cols:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)
                unseen_mask = ~input_data[col].isin(label_encoders[col].classes_)
                input_data.loc[unseen_mask, col] = 'unknown'
                if 'unknown' not in label_encoders[col].classes_:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        
        # Drop low importance features
        input_data = input_data.drop(columns=low_importance_features, errors='ignore')
        
        # Define expected features
        expected_features = [col for col in nsl_kdd_columns if col not in low_importance_features + ['class']]
        
        # Ensure all expected features are present
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns
        input_data = input_data[expected_features]
        
        # Scale features
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
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
    
    # Basic features
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
    
    # Map service
    service_map = {
        '80': 'http', '443': 'http', '21': 'ftp', '22': 'ssh', '23': 'telnet',
        '25': 'smtp', '53': 'dns', '110': 'pop3', '143': 'imap'
    }
    features['service'] = service_map.get(features['service'], 'other')
    
    # Connection-based features
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
    
    # Title
    elements.append(Paragraph("AI-Enhanced IDPS Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Summary
    summary_text = f"""
    Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    PCAP File: {filename}<br/>
    Total Packets Analyzed: {total_packets}<br/>
    Intrusions Detected: {intrusion_count}<br/>
    Normal Traffic: {total_packets - intrusion_count}
    """
    elements.append(Paragraph(summary_text, styles['BodyText']))
    elements.append(Spacer(1, 12))
    
    # Intrusion Timeline
    timeline_path = os.path.join(temp_dir, "intrusion_timeline.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([1 if r['is_intrusion'] else 0 for r in analysis_results], 'b-', label='Intrusion (1) / Normal (0)')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Packet Number")
    ax.set_ylabel("Status")
    ax.set_title("Intrusion Detection Timeline")
    ax.legend()
    fig.savefig(timeline_path, bbox_inches='tight')
    plt.close(fig)
    elements.append(ReportLabImage(timeline_path, width=400, height=200))
    elements.append(Spacer(1, 12))
    
    # Source IP Distribution
    src_ip_counts = pd.Series([r['src_ip'] for r in analysis_results if r['is_intrusion']]).value_counts().head(5)
    if not src_ip_counts.empty:
        ip_plot_path = os.path.join(temp_dir, "src_ip_distribution.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        src_ip_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel("Source IP")
        ax.set_ylabel("Intrusion Count")
        ax.set_title("Top 5 Source IPs with Intrusions")
        fig.savefig(ip_plot_path, bbox_inches='tight')
        plt.close(fig)
        elements.append(ReportLabImage(ip_plot_path, width=400, height=200))
        elements.append(Spacer(1, 12))
    
    # Results Table
    elements.append(Paragraph("Detailed Results", styles['Heading2']))
    data = [['Packet', 'Source IP', 'Prediction', 'Confidence', 'Anomaly']]
    for r in analysis_results[:10]:  # Limit to first 10 for brevity
        data.append([
            str(r['packet_num']),
            r['src_ip'],
            r['prediction'],
            f"{r['confidence']:.2%}",
            'Yes' if r['is_anomaly'] else 'No'
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
    
    # Recommendations
    elements.append(Paragraph("Recommendations", styles['Heading2']))
    rec_text = "Based on the analysis:<br/>"
    if intrusion_count > 0:
        rec_text += f"- Block the top source IPs: {', '.join(src_ip_counts.index)}.<br/>"
        rec_text += "- Review network security policies to mitigate detected attack types.<br/>"
    else:
        rec_text += "- No intrusions detected; continue monitoring with regular PCAP uploads."
    elements.append(Paragraph(rec_text, styles['BodyText']))
    
    # Build PDF
    doc.build(elements)
    return pdf_path

def select_uncertain_packets(analysis_results, uncertainty_threshold=0.1):
    return [i for i, r in enumerate(analysis_results) if abs(r['confidence'] - 0.5) < uncertainty_threshold]

def show_pcap_analysis():
    global model, scaler, label_encoders, le_class, autoencoder, model_type
    
    st.title("PCAP File Analysis")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    if not SCAPY_AVAILABLE:
        st.error("Scapy is not installed. Please install it with 'pip install scapy' and ensure libpcap is available.")
        return
    
    st.markdown("""
    ### PCAP File Analysis
    Upload a PCAP file to analyze network traffic for intrusions using AI-enhanced detection (LSTM, Autoencoders, Active Learning).
    """)
    
    # Alert settings
    st.subheader("Alert Settings")
    alert_threshold = st.slider("Alert Confidence Threshold", 0.5, 0.9, 0.7, 0.05)
    alert_recipient = st.text_input("Alert Recipient Email (Simulated)", "admin@example.com")
    
    # PCAP upload
    pcap_file = st.file_uploader("Upload PCAP File", type=["pcap", "pcapng"])
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
    
    if pcap_file is not None and st.button("Analyze PCAP"):
        with st.spinner("Analyzing PCAP file..."):
            try:
                # Save uploaded file
                temp_pcap = "temp_upload.pcap"
                with open(temp_pcap, "wb") as f:
                    f.write(pcap_file.read())
                
                # Read and process PCAP
                packets = rdpcap(temp_pcap)
                analysis_results = []
                intrusion_count = 0
                
                status_placeholder = st.empty()
                log_placeholder = st.empty()
                
                for i, packet in enumerate(packets):
                    features = process_packet(packet, st.session_state.connection_tracker)
                    if features is None:
                        st.write(f"Packet {i+1}: Skipping non-IP packet")
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
                    
                    if is_intrusion:
                        intrusion_count += 1
                        if confidence >= alert_threshold:
                            alert_message = f"Intrusion detected in packet {i+1} ({prediction}) from {src_ip} with confidence {confidence:.2%}"
                            st.session_state.alert_log.append({
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'message': alert_message,
                                'recipient': alert_recipient
                            })
                    
                    analysis_results.append({
                        'packet_num': i+1,
                        'is_intrusion': is_intrusion,
                        'prediction': prediction,
                        'confidence': confidence,
                        'src_ip': src_ip,
                        'is_anomaly': is_anomaly
                    })
                
                # Save to history
                history_entry = {
                    'timestamp': datetime.now(),
                    'filename': pcap_file.name,
                    'total_packets': len(packets),
                    'intrusion_count': intrusion_count,
                    'results': analysis_results
                }
                st.session_state.analysis_history.append(history_entry)
                
                # Update status
                status_placeholder.markdown(f"""
                **Analysis complete**  
                Total packets: {len(packets)}  
                Intrusions detected: {intrusion_count}  
                Normal traffic: {len(packets) - intrusion_count}
                """)
                
                # Display results
                st.subheader("Detailed Results")
                result_df = pd.DataFrame(analysis_results)
                st.dataframe(result_df[['packet_num', 'src_ip', 'prediction', 'confidence', 'is_anomaly']])
                
                # Feedback with Active Learning
                st.subheader("Provide Feedback")
                with st.form(f"feedback_form_{pcap_file.name}"):
                    uncertain_indices = select_uncertain_packets(analysis_results)
                    feedback_indices = st.multiselect(
                        "Select packets to mark as incorrect (suggested uncertain packets pre-selected)",
                        options=result_df.index,
                        default=uncertain_indices,
                        format_func=lambda x: f"Packet {result_df.loc[x, 'packet_num']} ({result_df.loc[x, 'prediction']}, Confidence: {result_df.loc[x, 'confidence']:.2%})"
                    )
                    correct_label = st.selectbox("Correct Label", ['normal'] + list(le_class.classes_))
                    feedback_submit = st.form_submit_button("Submit Feedback")
                    
                    if feedback_submit and feedback_indices:
                        for idx in feedback_indices:
                            packet_features = process_packet(packets[result_df.loc[idx, 'packet_num'] - 1], st.session_state.connection_tracker)
                            if packet_features:
                                packet_features['class'] = correct_label
                                st.session_state.feedback_data.append(packet_features)
                        st.success(f"Feedback saved for {len(feedback_indices)} packets.")
                
                # Generate and download report
                st.subheader("Download Report")
                pdf_path = generate_pdf_report(analysis_results, intrusion_count, len(packets), pcap_file.name)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
                
                # Clean up
                os.remove(temp_pcap)
                for temp_file in ['intrusion_timeline.png', 'src_ip_distribution.png']:
                    temp_file_path = os.path.join("temp", temp_file)
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                
            except Exception as e:
                st.error(f"Error during PCAP analysis: {str(e)}")
                if os.path.exists(temp_pcap):
                    os.remove(temp_pcap)

def show_historical_analysis():
    st.title("Historical Analysis Dashboard")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Upload and analyze a PCAP file first.")
        return
    
    st.markdown("""
    ### Historical Analysis
    View trends and insights from past PCAP analyses.
    """)
    
    # Convert history to DataFrame
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
    
    # Date filter
    st.subheader("Filter by Date")
    min_date = min(history_df['Timestamp']).date()
    max_date = max(history_df['Timestamp']).date()
    start_date, end_date = st.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    filtered_df = history_df[
        (history_df['Timestamp'].dt.date >= start_date) &
        (history_df['Timestamp'].dt.date <= end_date)
    ]
    
    # Summary
    st.subheader("Summary")
    st.write(f"Total Analyses: {len(filtered_df)}")
    st.write(f"Total Packets: {filtered_df['Total Packets'].sum()}")
    st.write(f"Total Intrusions: {filtered_df['Intrusions'].sum()}")
    
    # Visualizations
    st.subheader("Intrusion Trends")
    fig, ax = plt.subplots(figsize=(10, 4))
    filtered_df.set_index('Timestamp')['Intrusions'].plot(ax=ax, label='Intrusions')
    filtered_df.set_index('Timestamp')['Normal'].plot(ax=ax, label='Normal')
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.set_title("Intrusion and Normal Traffic Over Time")
    ax.legend()
    st.pyplot(fig)
    
    # Top Source IPs
    st.subheader("Top Source IPs with Intrusions")
    all_results = []
    for h in st.session_state.analysis_history:
        if h['timestamp'].date() >= start_date and h['timestamp'].date() <= end_date:
            all_results.extend(h['results'])
    
    if all_results:
        src_ip_counts = pd.Series([r['src_ip'] for r in all_results if r['is_intrusion']]).value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        src_ip_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel("Source IP")
        ax.set_ylabel("Intrusion Count")
        ax.set_title("Top 5 Source IPs with Intrusions")
        st.pyplot(fig)
    
    # Attack Type Distribution
    st.subheader("Attack Type Distribution")
    attack_types = pd.Series([r['prediction'] for r in all_results if r['is_intrusion']]).value_counts()
    if not attack_types.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        attack_types.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title("Distribution of Attack Types")
        st.pyplot(fig)

def show_alert_log():
    st.title("Alert Log")
    
    if not st.session_state.alert_log:
        st.info("No alerts generated yet. Analyze a PCAP file with a high confidence threshold to trigger alerts.")
        return
    
    st.markdown("""
    ### Alert Log
    View simulated email alerts for detected intrusions.
    """)
    
    alert_df = pd.DataFrame(st.session_state.alert_log)
    st.dataframe(alert_df[['timestamp', 'message', 'recipient']])
    
    if st.button("Clear Alert Log"):
        st.session_state.alert_log = []
        st.success("Alert log cleared.")

def show_retrain_model():
    global model, scaler, label_encoders, le_class, model_type
    
    st.title("Retrain Model with Feedback")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    ### Retrain Model
    Use feedback from PCAP analyses to improve the model. Active learning prioritizes uncertain predictions.
    """)
    
    if not st.session_state.feedback_data:
        st.info("No feedback data available. Provide feedback on PCAP analysis results first.")
        return
    
    # Display feedback data
    st.subheader("Feedback Data")
    feedback_df = pd.DataFrame(st.session_state.feedback_data)
    st.dataframe(feedback_df[[col for col in nsl_kdd_columns if col in feedback_df.columns]])
    
    # Retraining parameters
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
                # Load original training data
                train_file = st.session_state.get('last_train_file')
                if train_file:
                    data = pd.read_csv(train_file, names=nsl_kdd_columns, header=None)
                else:
                    data = pd.DataFrame(columns=nsl_kdd_columns)
                
                # Append feedback data
                feedback_df = pd.DataFrame(st.session_state.feedback_data)
                data = pd.concat([data, feedback_df], ignore_index=True)
                
                # Preprocess
                data, label_encoders, le_class = preprocess_data(data, {}, None, is_train=True)
                
                # Separate features and labels
                X = data.drop('class', axis=1)
                y = data['class']
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                
                # Train model
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
                
                # Evaluate
                if model_type == 'LSTM':
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = (model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int).flatten()
                else:
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                target_names = [le_class.classes_[i] for i in unique_labels]
                report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                
                # Save model and objects
                joblib.dump(model, 'idps_model.pkl')
                joblib.dump(scaler, 'scaler.pkl')
                joblib.dump(label_encoders, 'label_encoders.pkl')
                joblib.dump(le_class, 'le_class.pkl')
                joblib.dump(model_type, 'model_type.pkl')
                
                # Display results
                st.success("Model retrained successfully!")
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                st.subheader("Classification Report")
                st.text(report)
                
            except Exception as e:
                st.error(f"Error during retraining: {str(e)}")

def show_home():
    global model, model_type
    
    st.title("AI-Enhanced Intrusion Detection and Prevention System")
    st.markdown("""
    Welcome to the AI-Enhanced IDPS application. This system uses advanced AI (XGBoost, LSTM, Autoencoders) 
    to detect and prevent network intrusions using the NSL-KDD dataset.
    
    ### AI Features:
    - **Deep Learning (LSTM)**: Captures temporal patterns in traffic.
    - **Anomaly Detection (Autoencoders)**: Flags zero-day attacks.
    - **Active Learning**: Optimizes feedback for model improvement.
    
    ### Other Features:
    - Train and test intrusion detection models
    - Analyze PCAP files for intrusions
    - View historical analysis trends
    - Retrain models with feedback
    - Receive simulated alert notifications
    - Export analysis reports as PDF
    
    ### Navigation:
    - Use the sidebar to access different functionalities
    """)
    
    if model is not None:
        st.success(f"A trained {model_type} model is loaded and ready for use!")
    else:
        st.warning("No trained model found. Please train a model first.")

def show_train_model():
    global model, scaler, label_encoders, le_class, autoencoder, model_type
    
    st.title("Train New IDPS Model")
    
    if model is not None:
        st.warning("A trained model already exists. Training a new model will overwrite the existing one.")
    
    st.markdown("""
    ### Training Instructions:
    1. Upload the training data file (KDDTrain+.csv)
    2. Select model type (XGBoost or LSTM)
    3. Configure model parameters
    4. Optionally train an autoencoder for anomaly detection
    5. Click 'Train Model' button
    """)
    
    # File uploader
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    
    # Model selection
    model_type = st.selectbox("Model Type", ["XGBoost", "LSTM"], index=0)
    
    # Model parameters
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
        params = {}  # LSTM parameters are fixed in train_lstm_model
    
    # Autoencoder option
    train_autoencoder_option = st.checkbox("Train Autoencoder for Anomaly Detection")
    
    if st.button("Train Model"):
        if train_file is not None:
            with st.spinner("Training model..."):
                try:
                    # Load and preprocess data
                    data = pd.read_csv(train_file, names=nsl_kdd_columns, header=None)
                    data, label_encoders, le_class = preprocess_data(data, {}, None, is_train=True)
                    
                    # Separate features and labels
                    X = data.drop('class', axis=1)
                    y = data['class']
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    
                    # Apply SMOTE
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    
                    # Train model
                    if model_type == "LSTM":
                        model = train_lstm_model(X_train, y_train, X_test, y_test)
                        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        y_pred = (model.predict(X_test_reshaped, verbose=0) > 0.5).astype(int).flatten()
                    else:
                        model = XGBClassifier(**params)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Evaluate
                    accuracy = accuracy_score(y_test, y_pred)
                    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                    target_names = [le_class.classes_[i] for i in unique_labels]
                    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                    
                    # Train autoencoder if selected
                    if train_autoencoder_option:
                        X_normal = X_train[y_train == le_class.transform(['normal'])[0]]
                        autoencoder = train_autoencoder(scaler.transform(X_normal))
                        joblib.dump(autoencoder, 'autoencoder.pkl')
                    
                    # Save model and objects
                    joblib.dump(model, 'idps_model.pkl')
                    joblib.dump(scaler, 'scaler.pkl')
                    joblib.dump(label_encoders, 'label_encoders.pkl')
                    joblib.dump(le_class, 'le_class.pkl')
                    joblib.dump(model_type, 'model_type.pkl')
                    
                    # Save train file for retraining
                    st.session_state['last_train_file'] = train_file
                    
                    # Display results
                    st.success(f"{model_type} model trained successfully!")
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Classification Report")
                    st.text(report)
                    
                    # Feature importance (XGBoost only)
                    if model_type == "XGBoost":
                        st.subheader("Feature Importance")
                        importances = model.feature_importances_
                        feature_names = X.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), ax=ax)
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please upload a training data file first.")

def show_test_model():
    global model, scaler, label_encoders, le_class, model_type
    
    st.title("Test IDPS Model")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    ### Testing Options:
    1. Upload test data file (KDDTest+.csv)
    2. Enter manual test data
    """)
    
    test_option = st.radio("Select testing method", ["Upload Test File", "Manual Input"])
    
    if test_option == "Upload Test File":
        test_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
        
        if test_file is not None and st.button("Run Test"):
            with st.spinner("Testing model..."):
                try:
                    # Load and preprocess test data
                    test_data = pd.read_csv(test_file, names=nsl_kdd_columns, header=None)
                    test_data, _, _ = preprocess_data(test_data, label_encoders, le_class, is_train=False)
                    
                    # Separate features and labels
                    X_test = test_data.drop('class', axis=1)
                    y_test = test_data['class']
                    
                    # Scale features
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Make predictions
                    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
                    if model_type == 'LSTM':
                        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
                        y_pred_prob = model.predict(X_test_reshaped, verbose=0)[:, 0]
                    else:
                        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
                    y_pred = (y_pred_prob >= threshold).astype(int)
                    
                    # Get unique labels
                    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
                    target_names = [le_class.classes_[i] for i in unique_labels]
                    
                    # Evaluate
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names)
                    
                    # Display results
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Classification Report")
                    st.text(report)
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    conf_matrix = pd.crosstab(
                        le_class.inverse_transform(y_test),
                        le_class.inverse_transform(y_pred),
                        rownames=['Actual'],
                        colnames=['Predicted']
                    )
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during testing: {str(e)}")
    
    else:  # Manual Input
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
                    st.subheader("Detection Result")
                    if prediction == 'normal':
                        st.success(f"Normal traffic detected (Confidence: {confidence:.2%})")
                    else:
                        st.error(f"Intrusion detected: {prediction} (Confidence: {confidence:.2%})")
                    
                    st.subheader("Recommended Action")
                    if prediction == 'normal':
                        st.info("✅ Allow traffic - No malicious activity detected")
                    else:
                        st.warning("⛔ Block traffic - Potential intrusion detected")

def show_realtime_detection():
    global model, scaler, label_encoders, le_class, model_type
    
    st.title("Real-time Intrusion Detection Simulation")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    ### Real-time Simulation
    Simulate real-time network traffic monitoring using random data.
    """)
    
    st.subheader("Simulation Controls")
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
            results.append(is_intrusion)
            
            status_placeholder.markdown(f"""
            **Processing sample {i+1}/{num_samples}**  
            Last detection: {'Intrusion' if is_intrusion else 'Normal'}  
            Total intrusions detected: {intrusion_count}
            """)
            
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(results, 'b-', label='Intrusion (1) / Normal (0)')
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Status")
            ax.set_title("Detection Results Over Time")
            chart_placeholder.pyplot(fig)
            
            if is_intrusion:
                log_placeholder.markdown(f"""
                **Sample {i+1}**: 🚨 Intrusion detected ({prediction}) with confidence {confidence:.2%}
                """, unsafe_allow_html=True)
            else:
                log_placeholder.markdown(f"""
                **Sample {i+1}**: ✅ Normal traffic with confidence {confidence:.2%}
                """, unsafe_allow_html=True)
            
            time.sleep(delay)
        
        st.success(f"Simulation complete! Detected {intrusion_count} intrusions out of {num_samples} samples.")

def show_about():
    st.title("About This Project")
    
    st.markdown("""
    ### AI-Enhanced Intrusion Detection and Prevention System (IDPS)
    
    This application demonstrates advanced AI techniques for network intrusion detection and prevention.
    
    **AI Features:**
    - **Deep Learning (LSTM)**: Models temporal patterns in network traffic.
    - **Anomaly Detection (Autoencoders)**: Detects zero-day attacks via unsupervised learning.
    - **Active Learning**: Optimizes feedback by prioritizing uncertain predictions.
    
    **Other Features:**
    - Uses XGBoost or LSTM classifiers
    - Trained on the NSL-KDD dataset
    - Analyzes PCAP files for intrusions
    - Provides historical analysis dashboard
    - Supports model retraining with feedback
    - Simulates alert notifications
    - Exports analysis reports as PDF
    
    **Technical Details:**
    - Python 3.x
    - Streamlit for the web interface
    - Scikit-learn, TensorFlow for machine learning
    - XGBoost for traditional ML
    - Imbalanced-learn for handling class imbalance
    - Scapy for PCAP processing
    - ReportLab for PDF generation
    
    **Dataset Information:**
    The NSL-KDD dataset is a refined version of the KDD Cup 99 dataset, containing network connection records
    labeled as either normal or as specific attack types.
    
    **Attack Types:**
    - Denial of Service (DoS)
    - Probe (surveillance and probing)
    - Remote-to-Local (R2L)
    - User-to-Root (U2R)
    """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. **Train Model**: Upload training data, select XGBoost or LSTM, and optionally train an autoencoder
    2. **Test Model**: Evaluate the model with test data or manual input
    3. **PCAP Analysis**: Upload PCAP files to detect intrusions with anomaly detection
    4. **Historical Analysis**: View trends from past analyses
    5. **Retrain Model**: Improve the model with active learning feedback
    6. **Alert Log**: Review simulated intrusion alerts
    """)
    
    st.subheader("Disclaimer")
    st.markdown("""
    This is a demonstration project for educational purposes. Not intended for production use without security reviews.
    Deployed on Streamlit Cloud with TensorFlow CPU for compatibility.
    """)

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Home", "Train Model", "Test Model", "Real-time Detection", 
                                    "PCAP Analysis", "Historical Analysis", "Alert Log", "Retrain Model", "About"])
    
    if app_mode == "Home":
        show_home()
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
    elif app_mode == "About":
        show_about()

if __name__ == "__main__":
    main()
