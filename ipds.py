import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from geopy.distance import geodesic
from sklearn.ensemble import IsolationForest
import base64
import io
import sys
import sqlite3
import re
import bcrypt
import os
import random
import json

logging.basicConfig(level=logging.INFO, filename='atc_idps_sim.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# --- Suricata Rules (Simulated) ---
SURICATA_RULES = [
    {
        "sid": 1000001,
        "action": "alert",
        "protocol": "tcp",
        "src_ip": "any",
        "src_port": "any",
        "dst_ip": "$ATC_SERVERS",
        "dst_port": "22",
        "msg": "SSH Brute Force Attempt",
        "content": "SSH",
        "threshold": "limit 5, 60"
    },
    {
        "sid": 1000002,
        "action": "alert",
        "protocol": "udp",
        "src_ip": "any",
        "src_port": "any",
        "dst_ip": "$RADAR_FEED",
        "dst_port": "5000",
        "msg": "Radar Flood Detected",
        "content": "RDR",
        "rate": 100
    },
    {
        "sid": 1000003,
        "action": "drop",
        "protocol": "tcp",
        "src_ip": "any",
        "src_port": "any",
        "dst_ip": "$ATC_SERVERS",
        "dst_port": "80",
        "msg": "Unauthorized HTTP to ATC",
        "content": "GET /admin"
    },
    {
        "sid": 1000004,
        "action": "alert",
        "protocol": "udp",
        "src_ip": "any",
        "src_port": "any",
        "dst_ip": "$ADS_B",
        "dst_port": "30003",
        "msg": "ADS-B Spoofing Pattern",
        "content": "MSG,3",
        "pcre": "/callsign.*GHOST/"
    }
]

# --- Simulated ATC IPs ---
ATC_CONFIG = {
    "$ATC_SERVERS": ["10.10.10.1", "10.10.10.2"],
    "$RADAR_FEED": ["172.16.1.100"],
    "$ADS_B": ["192.168.1.50"]
}

def init_db():
    db_path = 'atc_users.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info("Old DB deleted")
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (username TEXT PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
            c.execute('INSERT OR IGNORE INTO users (username, email, password) VALUES (?, ?, ?)',
                      ('guardian', 'admin@atcguard.com', bcrypt.hashpw('admin'.encode(), bcrypt.gensalt()).decode()))
            conn.commit()
    except Exception as e:
        st.error("DB init failed")

def is_valid_email(email):
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email)

def is_valid_password(password):
    return len(password) >= 8 and any(c.isupper() for c in password) and any(c.isdigit() for c in password)

WICKET_THEME = {
    "primary_bg": "#0A0F2D",
    "card_bg": "rgba(30, 42, 68, 0.5)",
    "accent": "#00D4FF",
    "error": "#FF4D4D",
    "success": "#00FF99",
    "text_light": "#FFFFFF"
}

def apply_css():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono&display=swap');
            .stApp {{background: url('https://github.com/J4yd33n/IDPS-with-ML/blob/main/airplane.jpg?raw=true') cover center #0A0F2D; color: #E6E6FA; font-family: 'Roboto Mono';}}
            .stApp::before {{content:''; position:absolute; top:0; left:0; width:100%; height:100%; background:{WICKET_THEME['card_bg']}; z-index:-1;}}
            .card {{background:{WICKET_THEME['card_bg']}; border-radius:16px; padding:20px; margin:10px 0; box-shadow:0 0 20px rgba(0,212,255,0.2);}}
            .stButton>button {{background:linear-gradient(45deg,#00D4FF,#FF00FF); color:#0A0F2D; border-radius:25px; padding:12px 30px; font-family:'Orbitron';}}
            h1,h2,h3 {{font-family:'Orbitron'; color:#FFF; text-shadow:0 0 8px #00D4FF;}}
            .logo {{width:200px; margin:0 auto 20px; display:block;}}
        </style>
    """, unsafe_allow_html=True)

# --- Session State ---
for key in ['authenticated', 'alert_log', 'network_traffic', 'adsb_data', 'radar_data', 'acars_data', 'blocked_ips', 'quarantine_mode', 'panel_state']:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['authenticated', 'quarantine_mode'] else [] if 'data' in key or 'log' in key else set() if 'blocked' in key else 'sign_in'

init_db()

def render_auth():
    st.markdown('<div style="text-align:center;"><img src="https://github.com/J4yd33n/IDPS-with-ML/blob/main/FullLogo.jpg?raw=true" class="logo"></div>', unsafe_allow_html=True)
    if st.session_state.panel_state == 'sign_in':
        with st.form("login"):
            st.markdown("<h2 style='text-align:center;'>Sign In</h2>", unsafe_allow_html=True)
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In"):
                with sqlite3.connect('atc_users.db') as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT password FROM users WHERE username=?", (u,))
                    row = cur.fetchone()
                    if row and bcrypt.checkpw(p.encode(), row[0].encode()):
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid login")
        st.button("Sign Up", on_click=lambda: st.session_state.update(panel_state='sign_up'))
    else:
        with st.form("signup"):
            st.markdown("<h2 style='text-align:center;'>Sign Up</h2>", unsafe_allow_html=True)
            u = st.text_input("Username", key="su_u")
            e = st.text_input("Email")
            p = st.text_input("Password", type="password", key="su_p")
            if st.form_submit_button("Create"):
                if u == 'guardian': st.error("Reserved")
                elif not is_valid_email(e): st.error("Bad email")
                elif not is_valid_password(p): st.error("Weak password")
                else:
                    try:
                        with sqlite3.connect('atc_users.db') as conn:
                            conn.execute("INSERT INTO users VALUES (?,?,?)", (u, e, bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()))
                        st.success("Created")
                        st.session_state.panel_state = 'sign_in'
                        st.rerun()
                    except: st.error("Exists")
        st.button("Sign In", on_click=lambda: st.session_state.update(panel_state='sign_in'))

# --- Simulated Traffic ---
def gen_traffic(n=200):
    traffic = []
    for _ in range(n):
        src = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        dst = random.choice(ATC_CONFIG["$ATC_SERVERS"] + ATC_CONFIG["$RADAR_FEED"] + ATC_CONFIG["$ADS_B"])
        proto = random.choice(["tcp", "udp", "ADS-B", "ACARS"])
        port = random.choice([22, 80, 5000, 30003])
        size = random.randint(64, 1500)
        content = random.choice(["SSH", "GET /admin", "RDR", "MSG,3,callsign,GHOST123", "CLR NIG123"])
        traffic.append({
            'ts': datetime.now() - timedelta(seconds=random.randint(0,300)),
            'src_ip': src,
            'dst_ip': dst,
            'proto': proto,
            'port': port,
            'size': size,
            'content': content
        })
    return traffic

# --- Suricata Engine (Simulated) ---
def run_suricata(traffic):
    alerts = []
    ip_count = {}
    for pkt in traffic:
        ip_count[pkt['src_ip']] = ip_count.get(pkt['src_ip'], 0) + 1
        for rule in SURICATA_RULES:
            if pkt['proto'].lower() != rule['protocol']: continue
            if pkt['dst_ip'] not in [i for v in ATC_CONFIG.values() for i in v]: continue
            if rule['dst_port'] != 'any' and pkt['port'] != rule['dst_port']: continue
            if 'content' in rule and rule['content'] not in pkt['content']: continue
            if 'pcre' in rule and not re.search(rule['pcre'], pkt['content']): continue
            if rule['sid'] == 1000001 and ip_count[pkt['src_ip']] > 5:
                alerts.append({"ts": pkt['ts'], "type": rule['msg'], "sev": "high", "sid": rule['sid']})
            elif rule['sid'] == 1000002 and ip_count[pkt['src_ip']] > 100:
                alerts.append({"ts": pkt['ts'], "type": rule['msg'], "sev": "high", "sid": rule['sid']})
            else:
                alerts.append({"ts": pkt['ts'], "type": rule['msg'], "sev": "critical" if 'drop' in rule['action'] else "high", "sid": rule['sid']})
            if 'drop' in rule['action']:
                st.session_state.blocked_ips.add(pkt['src_ip'])
    return alerts

# --- ML Anomaly ---
def ml_anomaly(df):
    X = df[['size']].values
    model = IsolationForest(contamination=0.1)
    preds = model.fit_predict(X)
    return len(df[preds == -1])

# --- Data Generators ---
def gen_adsb(n=30):
    return [{
        'ts': datetime.now(),
        'icao': f"{random.randint(0xA00000, 0xAFFFFF):06X}",
        'callsign': "GHOST123" if random.random() < 0.15 else f"FLY{random.randint(100,999)}",
        'lat': np.random.uniform(4,14),
        'lon': np.random.uniform(2,15),
        'alt': np.random.uniform(10000,40000)
    } for _ in range(n)]

def gen_radar(n=20):
    return [{'ts': datetime.now(), 'id': f"RAD{i:03d}", 'lat': np.random.uniform(4,14), 'lon': np.random.uniform(2,15), 'alt': np.random.uniform(10000,40000)} for i in range(n)]

def gen_acars(n=15):
    return [{'ts': datetime.now(), 'reg': f"N{random.randint(100,999)}NG", 'msg': random.choice(["CLR NIG123", "POS RPT", "HACKEDCMD"])} for _ in range(n)]

# --- Dashboard ---
def dashboard():
    st.markdown('<div class="card"><h1>ATC IDPS</h1></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Packets", len(st.session_state.network_traffic))
    c2.metric("Alerts", len(st.session_state.alert_log))
    c3.metric("Blocked", len(st.session_state.blocked_ips))
    if st.button("Simulate Traffic"):
        st.session_state.network_traffic = gen_traffic()
        df = pd.DataFrame(st.session_state.network_traffic)
        suri_alerts = run_suricata(st.session_state.network_traffic)
        ml_count = ml_anomaly(df)
        if ml_count: suri_alerts.append({"ts": datetime.now(), "type": "ML Anomaly", "sev": "medium", "sid": 0})
        st.session_state.alert_log.extend(suri_alerts)
        st.session_state.adsb_data = gen_adsb()
        st.session_state.radar_data = gen_radar()
        st.session_state.acars_data = gen_acars()
        st.success("Done")
    if st.session_state.alert_log:
        st.subheader("Alerts")
        st.dataframe(pd.DataFrame(st.session_state.alert_log[-10:])[["ts","type","sev"]])

def network_map():
    if not st.session_state.network_traffic: 
        st.warning("Simulate first")
        return
    df = pd.DataFrame(st.session_state.network_traffic)
    fig = go.Figure(go.Scatter(x=df['ts'], y=df['size'], mode='markers', marker=dict(color='red' if df['src_ip'].iloc[0] in st.session_state.blocked_ips else 'cyan')))
    fig.update_layout(title="Traffic", paper_bgcolor=WICKET_THEME['card_bg'])
    st.plotly_chart(fig)

def adsb_map():
    if not st.session_state.adsb_data: 
        st.warning("Simulate")
        return
    df = pd.DataFrame(st.session_state.adsb_data)
    fig = go.Figure()
    for spoof in df['callsign'].str.contains("GHOST"):
        sub = df[spoof] if spoof else df[~df['callsign'].str.contains("GHOST")]
        fig.add_trace(go.Scattermapbox(lat=sub['lat'], lon=sub['lon'], marker=dict(color=WICKET_THEME['error'] if spoof else WICKET_THEME['success'])))
    fig.update_layout(mapbox=dict(style='open-street-map', center=dict(lat=9,lon=7), zoom=6))
    st.plotly_chart(fig)

def response():
    st.markdown('<div class="card"><h2>Response</h2></div>', unsafe_allow_html=True)
    if st.button("Quarantine Toggle"):
        st.session_state.quarantine_mode = not st.session_state.quarantine_mode
        st.rerun()
    ips = st.multiselect("Blocked IPs", list(st.session_state.blocked_ips))
    if st.button("Unblock"):
        for ip in ips: st.session_state.blocked_ips.discard(ip)
        st.rerun()

def rules_page():
    st.markdown('<div class="card"><h2>Suricata Rules</h2></div>', unsafe_allow_html=True)
    for r in SURICATA_RULES:
        st.code(f"{r['action']} {r['protocol']} {r['src_ip']} {r['src_port']} -> {r['dst_ip']} {r['dst_port']} (msg:\"{r['msg']}\"; sid:{r['sid']};)", language="text")

def main():
    apply_css()
    if not st.session_state.authenticated:
        render_auth()
        return
    st.sidebar.image("https://github.com/J4yd33n/IDPS-with-ML/blob/main/FullLogo.jpg?raw=true", use_column_width=True)
    page = st.sidebar.selectbox("Menu", ["Dashboard", "Network", "ADS-B", "Radar", "ACARS", "Response", "Suricata Rules"])
    if page == "Dashboard": dashboard()
    elif page == "Network": network_map()
    elif page == "ADS-B": adsb_map()
    elif page == "Radar": st.dataframe(pd.DataFrame(st.session_state.radar_data)) if st.session_state.radar_data else st.warning("Simulate")
    elif page == "ACARS": st.dataframe(pd.DataFrame(st.session_state.acars_data)) if st.session_state.acars_data else st.warning("Simulate")
    elif page == "Response": response()
    elif page == "Suricata Rules": rules_page()

if __name__ == "__main__":
    main()
