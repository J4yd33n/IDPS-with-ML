import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import nmap
import redis
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from flask import Flask
from models.idps_model import predict_traffic, train_model
from models.preprocess import preprocess_data
from scans.nmap_scan import run_nmap_scan
from scans.aviation_protocols import simulate_aviation_traffic
from utils.auth import authenticate_user
from utils.pdf_report import generate_nama_report
from utils.email_alerts import send_email_alert
from utils.audit_log import log_action
from utils.geo_viz import plot_threat_map
import config

# Initialize Flask for API and Dash for UI
server = Flask(__name__)
app = dash.Dash(__name__, server=server, title="NAMA IDPS")
app.config.suppress_callback_exceptions = True

# Initialize Redis for caching
redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)

# Load model and preprocessing objects
model, scaler, label_encoders, le_class, autoencoder, model_type = joblib.load('models/idps_model.pkl'), \
    joblib.load('models/scaler.pkl'), joblib.load('models/label_encoders.pkl'), \
    joblib.load('models/le_class.pkl'), joblib.load('models/autoencoder.pkl'), 'XGBoost'

# NAMA-specific columns for NSL-KDD dataset
nsl_kdd_columns = config.NSL_KDD_COLUMNS
categorical_cols = ['protocol_type', 'service', 'flag']
low_importance_features = config.LOW_IMPORTANCE_FEATURES

# Layout
app.layout = html.Div([
    # Header with NAMA branding
    html.Div([
        html.Img(src="/static/nama_logo.png", style={'height': '80px'}),
        html.H1("NAMA AI-Enhanced Intrusion Detection & Prevention System", style={'color': config.THEME['text']}),
        html.Button("Toggle Theme", id="theme-toggle", n_clicks=0),
        dcc.Store(id='theme-store', data='Light')
    ], style={'backgroundColor': config.THEME['background'], 'padding': '20px'}),

    # Authentication
    html.Div([
        dcc.Input(id="username", placeholder="Username", type="text"),
        dcc.Input(id="password", placeholder="Password", type="password"),
        html.Button("Login", id="login-button", n_clicks=0),
        html.Div(id="auth-message")
    ], id="auth-section", style={'display': 'block'}),

    # Main content (hidden until authenticated)
    html.Div([
        dcc.Tabs(id="tabs", value="home", children=[
            dcc.Tab(label="Home", value="home"),
            dcc.Tab(label="Train Model", value="train"),
            dcc.Tab(label="Test Model", value="test"),
            dcc.Tab(label="Real-time Detection", value="realtime"),
            dcc.Tab(label="NMAP Analysis", value="nmap"),
            dcc.Tab(label="ATC Monitoring", value="atc"),
            dcc.Tab(label="Compliance Dashboard", value="compliance"),
            dcc.Tab(label="Historical Analysis", value="history"),
            dcc.Tab(label="Alert Log", value="alerts"),
            dcc.Tab(label="Threat Intelligence", value="threats"),
            dcc.Tab(label="Documentation", value="docs")
        ]),
        html.Div(id="tab-content")
    ], id="main-content", style={'display': 'none'})
], style={'fontFamily': 'Arial', 'backgroundColor': config.THEME['background']})

# Callbacks
@app.callback(
    [Output("auth-section", "style"), Output("main-content", "style"), Output("auth-message", "children")],
    [Input("login-button", "n_clicks")],
    [State("username", "value"), State("password", "value")]
)
def login(n_clicks, username, password):
    if n_clicks > 0:
        if authenticate_user(username, password):
            log_action(username, "User logged in")
            return {'display': 'none'}, {'display': 'block'}, "Login successful!"
        return {'display': 'block'}, {'display': 'none'}, "Invalid credentials."
    return {'display': 'block'}, {'display': 'none'}, ""

@app.callback(
    Output("theme-store", "data"),
    [Input("theme-toggle", "n_clicks")],
    [State("theme-store", "data")]
)
def toggle_theme(n_clicks, current_theme):
    return "Dark" if current_theme == "Light" else "Light"

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value")]
)
def render_tab(tab):
    if tab == "home":
        return html.Div([
            html.H2("Welcome to NAMA IDPS"),
            html.P("Protect Nigeria's airspace with AI-driven intrusion detection and prevention."),
            html.Img(src="/static/nigeria_airspace_map.jpg", style={'width': '100%'}),
            html.H3("Key Features"),
            html.Ul([
                html.Li("Real-time NMAP scanning for network security"),
                html.Li("Monitoring of ATC protocols (ADS-B, ACARS)"),
                html.Li("ICAO/NCAA compliance tracking"),
                html.Li("Geo-location of threats with Nigerian airport focus")
            ])
        ])
    elif tab == "nmap":
        return html.Div([
            html.H2("Real-time NMAP Analysis"),
            dcc.Input(id="nmap-target", placeholder="Target IP/Hostname", type="text", value="192.168.1.1"),
            dcc.Dropdown(id="nmap-scan-type", options=[
                {'label': 'TCP SYN', 'value': 'TCP SYN'},
                {'label': 'TCP Connect', 'value': 'TCP Connect'},
                {'label': 'UDP', 'value': 'UDP'}
            ], value="TCP SYN"),
            dcc.Input(id="nmap-port-range", placeholder="Port Range (e.g., 1-1000)", type="text", value="1-1000"),
            html.Button("Run Scan", id="nmap-scan-button", n_clicks=0),
            html.Div(id="nmap-results"),
            dcc.Graph(id="nmap-chart")
        ])
    elif tab == "atc":
        return html.Div([
            html.H2("ATC Network Monitoring"),
            html.P("Monitor aviation-specific protocols for NAMA's network."),
            html.Button("Simulate ATC Traffic", id="atc-simulate", n_clicks=0),
            html.Div(id="atc-results"),
            dcc.Graph(id="atc-chart")
        ])
    elif tab == "compliance":
        return html.Div([
            html.H2("NCAA/ICAO Compliance Dashboard"),
            html.P("Track cybersecurity compliance for NAMA operations."),
            dcc.Graph(id="compliance-chart"),
            html.Button("Generate Compliance Report", id="compliance-report", n_clicks=0),
            html.Div(id="compliance-report-output")
        ])
    # Add other tab contents similarly...

@app.callback(
    [Output("nmap-results", "children"), Output("nmap-chart", "figure")],
    [Input("nmap-scan-button", "n_clicks")],
    [State("nmap-target", "value"), State("nmap-scan-type", "value"), State("nmap-port-range", "value")]
)
def run_nmap(n_clicks, target, scan_type, port_range):
    if n_clicks == 0:
        return "", px.bar()
    try:
        scan_results = run_nmap_scan(target, scan_type, port_range)
        df = pd.DataFrame(scan_results)
        open_ports = df[df['state'] == 'open']
        if open_ports.empty:
            return html.P("No open ports detected."), px.bar()
        
        table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in ['Port', 'Protocol', 'State', 'Service']])),
            html.Tbody([
                html.Tr([html.Td(open_ports.iloc[i][col]) for col in ['port', 'protocol', 'state', 'service']])
                for i in range(len(open_ports))
            ])
        ])
        fig = px.bar(open_ports, x='port', y='service', color='protocol', title=f"Open Ports on {target}")
        fig.update_layout(**config.THEME_LAYOUT)
        log_action("system", f"NMAP scan on {target}")
        return table, fig
    except Exception as e:
        return html.P(f"Error: {str(e)}"), px.bar()

@app.callback(
    [Output("atc-results", "children"), Output("atc-chart", "figure")],
    [Input("atc-simulate", "n_clicks")]
)
def simulate_atc(n_clicks):
    if n_clicks == 0:
        return "", px.line()
    atc_data = simulate_aviation_traffic()
    df = pd.DataFrame(atc_data)
    predictions = [predict_traffic(pd.DataFrame([row])) for row in atc_data]
    df['prediction'] = [p[0] for p in predictions]
    df['confidence'] = [p[1] for p in predictions]
    intrusions = df[df['prediction'] != 'normal']
    if not intrusions.empty:
        send_email_alert(config.EMAIL_RECIPIENT, "ATC Intrusion Detected", intrusions.to_string())
    fig = px.scatter(df, x='timestamp', y='confidence', color='prediction', title="ATC Traffic Analysis")
    fig.update_layout(**config.THEME_LAYOUT)
    return html.P(f"Detected {len(intrusions)} intrusions."), fig

@app.callback(
    Output("compliance-report-output", "children"),
    [Input("compliance-report", "n_clicks")]
)
def generate_compliance_report(n_clicks):
    if n_clicks == 0:
        return ""
    report_path = generate_nama_report()
    return html.A("Download Compliance Report", href=report_path, download="nama_compliance_report.pdf")

if __name__ == "__main__":
    app.run_server(debug=config.DEBUG)
