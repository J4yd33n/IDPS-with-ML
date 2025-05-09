import streamlit as st
   import streamlit.components.v1 as components
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

   # NSL-KDD columns (unchanged)
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

   # Wicket-inspired theme updated with new color palette
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
           </style>
       """
       st.markdown(css, unsafe_allow_html=True)

   # Simplified login HTML with Tailwind and basic animations
   LOGIN_HTML = """
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>NAMA IDPS Login</title>
       <script src="https://cdn.tailwindcss.com"></script>
       <style>
           @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700&family=Inter:wght@400;500&display=swap');
           .glassmorphic {
               background: rgba(255, 255, 255, 0.05);
               backdrop-filter: blur(10px);
               border: 1px solid rgba(255, 255, 255, 0.1);
           }
           .animate-shake {
               animation: shake 0.4s ease-in-out;
           }
           @keyframes shake {
               0%, 100% { transform: translateX(0); }
               20%, 60% { transform: translateX(-10px); }
               40%, 80% { transform: translateX(10px); }
           }
           .particles-bg {
               position: fixed;
               inset: 0;
               background: url('https://via.placeholder.com/1920x1080/1A1F36/1A1F36?text=Particles');
               z-index: -1;
           }
       </style>
   </head>
   <body class="bg-[#1A1F36] min-h-screen flex items-center justify-center">
       <div class="particles-bg"></div>
       <div id="login-card" class="glassmorphic p-8 rounded-xl max-w-md w-full">
           <h1 class="text-3xl font-bold text-white mb-6 text-center font-['Poppins']">NAMA IDPS</h1>
           <form id="login-form">
               <div class="mb-4">
                   <label class="block text-gray-300 mb-2 font-['Inter']" for="username">Username</label>
                   <input type="text" id="username" class="w-full p-3 bg-transparent border border-gray-600 rounded-lg text-white focus:border-[#3B82F6] focus:ring-2 focus:ring-[#3B82F6] transition-all font-['Inter']" autofocus required>
               </div>
               <div class="mb-6 relative">
                   <label class="block text-gray-300 mb-2 font-['Inter']" for="password">Password</label>
                   <input type="password" id="password" class="w-full p-3 bg-transparent border border-gray-600 rounded-lg text-white focus:border-[#3B82F6] focus:ring-2 focus:ring-[#3B82F6] transition-all font-['Inter']" required>
                   <button type="button" id="toggle-password" class="absolute right-3 top-10 text-gray-400 hover:text-white font-['Inter']">Show</button>
               </div>
               <p id="error-message" class="text-[#EF4444] mb-4 text-center font-['Inter'] hidden"></p>
               <button type="submit" class="w-full bg-[#3B82F6] text-white p-3 rounded-lg hover:bg-[#2563EB] transition-all font-['Poppins']">Login</button>
           </form>
       </div>
       <script>
           const form = document.getElementById('login-form');
           const errorMessage = document.getElementById('error-message');
           const loginCard = document.getElementById('login-card');
           const togglePassword = document.getElementById('toggle-password');
           const passwordInput = document.getElementById('password');
           togglePassword.addEventListener('click', () => {
               const type = passwordInput.type === 'password' ? 'text' : 'password';
               passwordInput.type = type;
               togglePassword.textContent = type === 'password' ? 'Show' : 'Hide';
           });
           form.addEventListener('submit', async (e) => {
               e.preventDefault();
               const username = document.getElementById('username').value;
               const password = document.getElementById('password').value;
               errorMessage.classList.add('hidden');
               try {
                   const response = await fetch('/api/login', {
                       method: 'POST',
                       headers: { 'Content-Type': 'application/json' },
                       body: JSON.stringify({ username, password })
                   });
                   if (!response.ok) throw new Error('Invalid credentials');
                   window.location.href = '/dashboard';
               } catch (err) {
                   errorMessage.textContent = err.message;
                   errorMessage.classList.remove('hidden');
                   loginCard.classList.add('animate-shake');
                   setTimeout(() => loginCard.classList.remove('animate-shake'), 400);
               }
           });
       </script>
   </body>
   </html>
   """

   # Initialize session state
   if 'analysis_history' not in st.session_state:
       st.session_state.analysis_history = []
   if 'alert_log' not in st.session_state:
       st.session_state.alert_log = []
   if 'authenticated' not in st.session_state:
       st.session_state.authenticated = False
   if 'compliance_metrics' not in st.session_state:
       st.session_state.compliance_metrics = {'detection_rate': 0, 'open_ports': 0, 'alerts': 0}
   if 'user_activity' not in st.session_state:
       st.session_state.user_activity = {}
   if 'equipment_status' not in st.session_state:
       st.session_state.equipment_status = []
   if 'theme_mode' not in st.session_state:
       st.session_state.theme_mode = 'dark'

   # User database setup (unchanged)
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
           return False
       conn = sqlite3.connect('nama_users.db')
       c = conn.cursor()
       c.execute("SELECT password FROM users WHERE username = ?", (username,))
       result = c.fetchone()
       conn.close()
       if result:
           stored_password = result[0]
           return bcrypt.checkpw(password.encode('utf-8'), stored_password)
       return False

   def log_user_activity(username, action):
       conn = sqlite3.connect('nama_users.db')
       c = conn.cursor()
       c.execute("INSERT INTO user_activity (username, timestamp, action) VALUES (?, ?, ?)",
                 (username, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action))
       conn.commit()
       conn.close()
       logger.info(f"User: {username}, Action: {action}")

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
           st.title("NAMA IDPS - Login")
           components.html(LOGIN_HTML, height=600, scrolling=True)
           return
       
       # Rest of the app (unchanged)
       st.sidebar.image("https://via.placeholder.com/120x60.png?text=NAMA+IDPS", use_column_width=True)
       st.sidebar.selectbox("Theme", ["Dark"], index=0)
       st.title("NAMA Intrusion Detection and Prevention System")
       st.markdown(f"Welcome, {st.session_state.username} | [Logout](#)", unsafe_allow_html=True)
       
       if st.button("Logout"):
           st.session_state.authenticated = False
           st.session_state.username = None
           log_user_activity(st.session_state.username, "Logged out")
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
       
       # Add other menu options (omitted for brevity, same as previous)
   
   if __name__ == "__main__":
       main()
