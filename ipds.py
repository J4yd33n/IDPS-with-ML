import pandas as pd
import numpy as np
import streamlit as st
import joblib
import time
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scapy.all import rdpcap, TCP, UDP, ICMP, IP, sniff, get_if_list
from scapy.arch.windows import IFACES
import subprocess
import io
import os
from datetime import datetime, timedelta
import platform
import ctypes
import sys

# Set page config must be first command
st.set_page_config(page_title="ML-Based IDPS", page_icon="🛡️", layout="wide")

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

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('idps_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        le_class = joblib.load('le_class.pkl')
        st.write("Model and preprocessing objects loaded successfully.")
        return model, scaler, label_encoders, le_class
    except Exception as e:
        st.error(f"Failed to load model or preprocessing objects: {str(e)}")
        return None, None, None, None

model, scaler, label_encoders, le_class = load_model()

def run_as_admin():
    """Relaunch the script as Administrator if not already elevated."""
    if platform.system() == "Windows" and not ctypes.windll.shell32.IsUserAnAdmin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()

def check_admin_privileges():
    """Check if the process is running with administrative privileges."""
    return ctypes.windll.shell32.IsUserAnAdmin() != 0

def check_npcap_installed():
    """Check if Npcap is installed and running."""
    if platform.system() == "Windows":
        try:
            # Check Npcap service
            result = subprocess.run("sc query npcap", capture_output=True, text=True, shell=True)
            if "RUNNING" not in result.stdout:
                return False, "Npcap service not running. Run 'net start npcap' or reinstall Npcap."
            # Check if Scapy can list interfaces
            interfaces = get_if_list()
            if not interfaces:
                return False, "No interfaces detected. Ensure Npcap is installed in WinPcap API-compatible mode."
            return True, "Npcap is installed and running."
        except Exception as e:
            return False, f"Npcap check failed: {str(e)}. Reinstall Npcap from https://npcap.com."
    return True, "Non-Windows system, assuming Npcap not required."

def get_friendly_interfaces():
    """Map Scapy interface GUIDs to friendly names, filter active interfaces."""
    interfaces = get_if_list()
    friendly_interfaces = []
    try:
        # Get adapter status via netsh
        result = subprocess.run("netsh interface show interface", capture_output=True, text=True, shell=True)
        active_interfaces = [line.split()[-1] for line in result.stdout.splitlines() if "Connected" in line]
        
        for iface in interfaces:
            try:
                iface_info = IFACES.dev_from_name(iface)
                friendly_name = iface_info.description if hasattr(iface_info, 'description') else iface
                # Check if interface is active (has IP address)
                ipconfig = subprocess.run("ipconfig", capture_output=True, text=True, shell=True)
                is_active = any(friendly_name in line for line in ipconfig.stdout.splitlines() if "IPv4 Address" in line)
                if is_active or any(friendly_name.lower().startswith(name.lower()) for name in active_interfaces):
                    friendly_interfaces.append((iface, friendly_name))
            except:
                friendly_interfaces.append((iface, iface))
    except Exception as e:
        st.warning(f"Could not filter active interfaces: {str(e)}. Showing all interfaces.")
        for iface in interfaces:
            friendly_interfaces.append((iface, iface))
    return friendly_interfaces

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
    
    return df, label_encoders, le_class

def predict_traffic(input_data, threshold=0.5):
    global model, scaler, label_encoders, le_class
    
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
        
        # Define expected features (based on training data after preprocessing)
        expected_features = [col for col in nsl_kdd_columns if col not in low_importance_features + ['class']]
        
        # Ensure all expected features are present
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match expected features
        input_data = input_data[expected_features]
        
        # Scale features
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        pred_prob = model.predict_proba(input_data_scaled)[:, 1]
        prediction = (pred_prob >= threshold).astype(int)
        prediction_label = le_class.inverse_transform(prediction)[0]
        
        return prediction_label, pred_prob[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def process_packet(packet, connection_tracker, time_window=10):
    """Process a single packet and extract NSL-KDD-like features."""
    features = {col: 0 for col in nsl_kdd_columns if col != 'class'}
    
    if IP not in packet:
        return None
    
    # Basic features
    features['protocol_type'] = packet[IP].proto
    if packet[IP].proto == 6:
        features['protocol_type'] = 'tcp'
        if TCP in packet:
            features['src_bytes'] = len(packet[TCP].payload)
            features['dst_bytes'] = 0  # Approximation, needs reverse traffic
            features['service'] = str(packet[TCP].dport)
            features['flag'] = 'SF'  # Simplified, based on TCP flags
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
    
    # Map service to common NSL-KDD services
    service_map = {
        '80': 'http', '443': 'http', '21': 'ftp', '22': 'ssh', '23': 'telnet',
        '25': 'smtp', '53': 'dns', '110': 'pop3', '143': 'imap'
    }
    features['service'] = service_map.get(features['service'], 'other')
    
    # Connection-based features (approximated)
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    dst_port = packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
    timestamp = packet.time
    
    # Track connections in a time window
    connection_key = (src_ip, dst_ip, dst_port, features['protocol_type'])
    current_time = datetime.fromtimestamp(timestamp)
    connection_tracker[connection_key] = [
        pkt for pkt in connection_tracker.get(connection_key, [])
        if datetime.fromtimestamp(pkt['time']) > current_time - timedelta(seconds=time_window)
    ]
    connection_tracker[connection_key].append({'time': timestamp, 'packet': packet})
    
    # Traffic features
    features['count'] = len(connection_tracker[connection_key])
    features['srv_count'] = len([
        pkt for pkt in connection_tracker.get((dst_ip, src_ip, dst_port, features['protocol_type']), [])
    ])
    features['same_srv_rate'] = features['count'] / max(1, features['srv_count'])
    features['diff_srv_rate'] = 1 - features['same_srv_rate']
    
    # Host-based features (approximated)
    features['dst_host_count'] = len(set(
        pkt['packet'][IP].src for pkt in connection_tracker.get((dst_ip, src_ip, dst_port, features['protocol_type']), [])
    ))
    features['dst_host_srv_count'] = features['srv_count']
    features['dst_host_same_srv_rate'] = features['same_srv_rate']
    
    return features

def show_live_monitoring():
    global model, scaler, label_encoders, le_class
    
    st.title("Live Network Monitoring")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    ### Live Network Traffic Monitoring
    Monitor real-time network traffic or upload a PCAP file for analysis.
    The system will analyze packets and flag potential intrusions.
    """)
    
    # Connection tracker for aggregating features
    if 'connection_tracker' not in st.session_state:
        st.session_state.connection_tracker = {}
    
    # Check prerequisites
    if not check_admin_privileges():
        st.error("Administrative privileges required. This script will attempt to relaunch as Administrator.")
        st.info("If prompted, click 'Yes' in the User Account Control (UAC) dialog. Alternatively, run manually: Right-click Command Prompt, select 'Run as administrator', and execute 'streamlit run ipds.py'.")
        if st.button("Relaunch as Administrator"):
            run_as_admin()
        return
    
    npcap_status, npcap_message = check_npcap_installed()
    if not npcap_status:
        st.error(f"Npcap error: {npcap_message}")
        st.info("Install Npcap from https://npcap.com (select WinPcap API-compatible mode). Verify service: 'sc query npcap'. Restart: 'net stop npcap && net start npcap'.")
        return
    
    st.success("Npcap is installed and running.")
    
    # Monitoring options
    monitoring_options = ["Live Capture (Scapy)", "Upload PCAP File"]
    monitoring_option = st.radio("Select monitoring method", monitoring_options)
    
    if monitoring_option == "Live Capture (Scapy)":
        st.info("Select the active network interface (e.g., Wi-Fi or Ethernet). Ensure network traffic is flowing (e.g., run 'ping 8.8.8.8').")
        
        # List available interfaces with friendly names
        try:
            friendly_interfaces = get_friendly_interfaces()
            if not friendly_interfaces:
                st.error("No active network interfaces found. Check network adapters ('ipconfig') and ensure Npcap is installed.")
                st.info("Alternative: Use Wireshark to capture packets, save as PCAP, and upload below.")
                return
            interface_options = [f"{friendly_name} ({iface})" for iface, friendly_name in friendly_interfaces]
            interface_selection = st.selectbox("Network Interface", interface_options, help="Select the active network interface with an IP address.")
            # Extract the GUID from selection
            selected_interface = next(iface for iface, _ in friendly_interfaces if iface in interface_selection)
        except Exception as e:
            st.error(f"Error listing interfaces: {str(e)}")
            st.info("Verify Npcap ('sc query npcap'), adapters ('netsh interface show interface'), and run as Administrator.")
            return
        
        num_packets = st.slider("Number of packets to capture", 10, 1000, 100)
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
        
        if st.button("Start Live Capture"):
            with st.spinner("Capturing network traffic..."):
                try:
                    # Capture packets using Scapy's sniff
                    packets = sniff(iface=selected_interface, count=num_packets, timeout=60)
                    if not packets:
                        st.warning("No packets captured. Generate traffic (e.g., 'ping 8.8.8.8' or 'curl http://example.com') and verify interface ('ipconfig').")
                        st.info("Alternative: Use Wireshark to capture packets, save as PCAP, and upload below.")
                    
                    results = []
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
                        if is_intrusion:
                            intrusion_count += 1
                            # Basic IPS: Block source IP using Windows Firewall
                            src_ip = packet[IP].src
                            try:
                                subprocess.run(
                                    f"netsh advfirewall firewall add rule name='Block {src_ip}' dir=in action=block remoteip={src_ip}",
                                    shell=True, check=True
                                )
                            except subprocess.CalledProcessError as e:
                                st.warning(f"Failed to add firewall rule for {src_ip}: {e}")
                        
                        results.append(is_intrusion)
                        
                        # Update status
                        status_placeholder.markdown(f"""
                        **Processing packet {i+1}/{len(packets)}**  
                        Last detection: {'Intrusion' if is_intrusion else 'Normal'}  
                        Total intrusions detected: {intrusion_count}
                        """)
                        
                        # Update log
                        if is_intrusion:
                            log_placeholder.markdown(f"""
                            **Packet {i+1}**: 🚨 Intrusion detected ({prediction}) with confidence {confidence:.2%}
                            Source IP: {packet[IP].src}
                            """, unsafe_allow_html=True)
                        else:
                            log_placeholder.markdown(f"""
                            **Packet {i+1}**: ✅ Normal traffic with confidence {confidence:.2%}
                            Source IP: {packet[IP].src}
                            """, unsafe_allow_html=True)
                        
                        time.sleep(0.1)  # Simulate real-time effect
                    
                    st.success(f"Capture complete! Detected {intrusion_count} intrusions out of {len(packets)} packets.")
                
                except PermissionError as e:
                    st.error(f"Permission error during live capture: {str(e)}")
                    st.info("Ensure you are running as Administrator. Relaunch via the button above or manually: Right-click Command Prompt, select 'Run as administrator', and execute 'streamlit run ipds.py'.")
                except Exception as e:
                    st.error(f"Error during live capture: {str(e)}")
                    st.info("Possible causes:\n"
                            "- Incorrect interface: Verify with 'ipconfig' and select the active adapter.\n"
                            "- Npcap issue: Reinstall from https://npcap.com (WinPcap API-compatible mode).\n"
                            "- Antivirus/Firewall: Disable temporarily or add exceptions for python.exe and streamlit.exe.\n"
                            "- No traffic: Generate traffic with 'ping 8.8.8.8' or 'curl http://example.com'.\n"
                            "Alternative: Use Wireshark to capture packets, save as PCAP, and upload below.")
    
    else:  # Upload PCAP File
        st.info("Upload a PCAP file captured with Wireshark or another tool if live capture is not working.")
        pcap_file = st.file_uploader("Upload PCAP File", type=["pcap", "pcapng"])
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
        
        if pcap_file is not None and st.button("Analyze PCAP"):
            with st.spinner("Analyzing PCAP file..."):
                try:
                    # Save uploaded file temporarily
                    temp_pcap = "temp_upload.pcap"
                    with open(temp_pcap, "wb") as f:
                        f.write(pcap_file.read())
                    
                    # Read and process PCAP
                    packets = rdpcap(temp_pcap)
                    results = []
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
                        if is_intrusion:
                            intrusion_count += 1
                            # Basic IPS: Block source IP
                            src_ip = packet[IP].src
                            try:
                                subprocess.run(
                                    f"netsh advfirewall firewall add rule name='Block {src_ip}' dir=in action=block remoteip={src_ip}",
                                    shell=True, check=True
                                )
                            except subprocess.CalledProcessError as e:
                                st.warning(f"Failed to add firewall rule for {src_ip}: {e}")
                        
                        results.append(is_intrusion)
                        
                        # Update status
                        status_placeholder.markdown(f"""
                        **Processing packet {i+1}/{len(packets)}**  
                        Last detection: {'Intrusion' if is_intrusion else 'Normal'}  
                        Total intrusions detected: {intrusion_count}
                        """)
                        
                        # Update log
                        if is_intrusion:
                            log_placeholder.markdown(f"""
                            **Packet {i+1}**: 🚨 Intrusion detected ({prediction}) with confidence {confidence:.2%}
                            Source IP: {packet[IP].src}
                            """, unsafe_allow_html=True)
                        else:
                            log_placeholder.markdown(f"""
                            **Packet {i+1}**: ✅ Normal traffic with confidence {confidence:.2%}
                            Source IP: {packet[IP].src}
                            """, unsafe_allow_html=True)
                        
                        time.sleep(0.1)  # Simulate real-time effect
                    
                    # Clean up
                    os.remove(temp_pcap)
                    st.success(f"Analysis complete! Detected {intrusion_count} intrusions out of {len(packets)} packets.")
                
                except Exception as e:
                    st.error(f"Error during PCAP analysis: {str(e)}")
                    if os.path.exists(temp_pcap):
                        os.remove(temp_pcap)

def show_home():
    global model
    
    st.title("Machine Learning-Based Intrusion Detection and Prevention System")
    st.markdown("""
    Welcome to the ML-Based IDPS application. This system uses XGBoost to detect and prevent network intrusions 
    using the NSL-KDD dataset.
    
    ### Features:
    - Train a new intrusion detection model
    - Test the model with sample data
    - Perform real-time intrusion detection
    - Monitor live network traffic
    - Visualize model performance
    
    ### Navigation:
    - Use the sidebar to access different functionalities
    """)
    
    if model is not None:
        st.success("A trained model is loaded and ready for use!")
    else:
        st.warning("No trained model found. Please train a model first.")
    
    # Display sample data
    st.subheader("Sample Network Traffic Data")
    sample_data = {
        'duration': [0],
        'protocol_type': ['tcp'],
        'service': ['http'],
        'flag': ['SF'],
        'src_bytes': [100],
        'dst_bytes': [200],
        'wrong_fragment': [0],
        'hot': [0],
        'logged_in': [1],
        'num_compromised': [0],
        'count': [2],
        'srv_count': [2],
        'serror_rate': [0.0],
        'srv_serror_rate': [0.0],
        'rerror_rate': [0.0],
        'srv_rerror_rate': [0.0],
        'same_srv_rate': [1.0],
        'diff_srv_rate': [0.0],
        'srv_diff_host_rate': [0.0],
        'dst_host_count': [2],
        'dst_host_srv_count': [2],
        'dst_host_same_srv_rate': [1.0],
        'dst_host_diff_srv_rate': [0.0],
        'dst_host_same_src_port_rate': [0.0],
        'dst_host_srv_diff_host_rate': [0.0],
        'dst_host_serror_rate': [0.0],
        'dst_host_srv_serror_rate': [0.0],
        'dst_host_rerror_rate': [0.0],
        'dst_host_srv_rerror_rate': [0.0]
    }
    st.json(sample_data)

def show_train_model():
    global model, scaler, label_encoders, le_class
    
    st.title("Train New IDPS Model")
    
    if model is not None:
        st.warning("A trained model already exists. Training a new model will overwrite the existing one.")
    
    st.markdown("""
    ### Training Instructions:
    1. Upload the training data file (KDDTrain+.csv)
    2. Configure model parameters (optional)
    3. Click 'Train Model' button
    """)
    
    # File uploader
    train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    
    # Model parameters
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Number of estimators", 50, 500, 200, 50)
        max_depth = st.slider("Max depth", 3, 10, 6)
    with col2:
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        scale_pos_weight = st.slider("Scale positive weight", 1, 5, 2)
    
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
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42
                    )
                    
                    model.fit(X_train, y_train)
                    
                    # Evaluate
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
                    
                    # Display results
                    st.success("Model trained successfully!")
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Classification Report")
                    st.text(report)
                    
                    # Feature importance
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
    global model, scaler, label_encoders, le_class
    
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
                    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
                    y_pred = (y_pred_prob >= threshold).astype(int)
                    
                    # Get unique labels in predictions and test data
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
        
        # Create input form
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
                # Create input DataFrame
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
                
                # Make prediction
                prediction, confidence = predict_traffic(input_df, threshold)
                
                if prediction is not None and confidence is not None:
                    # Display results
                    st.subheader("Detection Result")
                    if prediction == 'normal':
                        st.success(f"Normal traffic detected (Confidence: {confidence:.2%})")
                    else:
                        st.error(f"Intrusion detected: {prediction} (Confidence: {confidence:.2%})")
                    
                    # Show action recommendation
                    st.subheader("Recommended Action")
                    if prediction == 'normal':
                        st.info("✅ Allow traffic - No malicious activity detected")
                    else:
                        st.warning("⛔ Block traffic - Potential intrusion detected")

def show_realtime_detection():
    global model, scaler, label_encoders, le_class
    
    st.title("Real-time Intrusion Detection")
    
    if model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    st.markdown("""
    ### Real-time Monitoring:
    This page simulates real-time network traffic monitoring. 
    The system will analyze traffic and flag potential intrusions.
    """)
    
    # Simulation controls
    st.subheader("Simulation Controls")
    simulation_speed = st.select_slider("Simulation Speed", options=["Slow", "Normal", "Fast"], value="Normal")
    num_samples = st.slider("Number of samples to simulate", 10, 100, 20)
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.4, 0.05)
    
    if st.button("Start Simulation"):
        # Generate random test data for simulation
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
        
        # Create placeholder for live updates
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        log_placeholder = st.empty()
        
        # Initialize results
        results = []
        intrusion_count = 0
        
        # Determine speed
        delay = 1.0 if simulation_speed == "Slow" else 0.5 if simulation_speed == "Normal" else 0.1
        
        # Process each sample
        for i in range(num_samples):
            # Get current sample
            current_sample = sample_df.iloc[[i]]
            
            # Make prediction
            prediction, confidence = predict_traffic(current_sample, threshold)
            
            if prediction is None or confidence is None:
                continue
            
            # Record result
            is_intrusion = prediction != 'normal'
            if is_intrusion:
                intrusion_count += 1
            results.append(is_intrusion)
            
            # Update status
            status_placeholder.markdown(f"""
            **Processing sample {i+1}/{num_samples}**  
            Last detection: {'Intrusion' if is_intrusion else 'Normal'}  
            Total intrusions detected: {intrusion_count}
            """)
            
            # Update chart
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(results, 'b-', label='Intrusion (1) / Normal (0)')
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Status")
            ax.set_title("Detection Results Over Time")
            chart_placeholder.pyplot(fig)
            
            # Update log
            if is_intrusion:
                log_placeholder.markdown(f"""
                **Sample {i+1}**: 🚨 Intrusion detected ({prediction}) with confidence {confidence:.2%}
                """, unsafe_allow_html=True)
            else:
                log_placeholder.markdown(f"""
                **Sample {i+1}**: ✅ Normal traffic with confidence {confidence:.2%}
                """, unsafe_allow_html=True)
            
            # Add delay for simulation effect
            time.sleep(delay)
        
        # Final summary
        st.success(f"Simulation complete! Detected {intrusion_count} intrusions out of {num_samples} samples.")

def show_about():
    st.title("About This Project")
    
    st.markdown("""
    ### Machine Learning-Based Intrusion Detection and Prevention System (IDPS)
    
    This application demonstrates how machine learning can be used to detect and prevent network intrusions.
    
    **Key Features:**
    - Uses XGBoost classifier for intrusion detection
    - Trained on the NSL-KDD dataset
    - Provides real-time monitoring capabilities
    - Monitors live network traffic using Scapy
    - Offers configurable detection threshold
    
    **Technical Details:**
    - Python 3.x
    - Streamlit for the web interface
    - Scikit-learn for preprocessing and evaluation
    - XGBoost for the machine learning model
    - Imbalanced-learn for handling class imbalance
    - Scapy for packet processing
    
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
    1. **Train Model**: Upload training data and train a new model
    2. **Test Model**: Evaluate the model with test data or manual input
    3. **Real-time Detection**: Simulate real-time network monitoring
    4. **Live Network Monitoring**: Capture and analyze live traffic or upload PCAP files
    """)
    
    st.subheader("Disclaimer")
    st.markdown("""
    This is a demonstration project for educational purposes only. 
    Not intended for production use without proper security reviews and enhancements.
    """)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Home", "Train Model", "Test Model", "Real-time Detection", "Live Network Monitoring", "About"])
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Train Model":
        show_train_model()
    elif app_mode == "Test Model":
        show_test_model()
    elif app_mode == "Real-time Detection":
        show_realtime_detection()
    elif app_mode == "Live Network Monitoring":
        show_live_monitoring()
    elif app_mode == "About":
        show_about()

if __name__ == "__main__":
    run_as_admin()
    main()
