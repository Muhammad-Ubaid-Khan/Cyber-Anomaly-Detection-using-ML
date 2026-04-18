import streamlit as st
import pandas as pd
import joblib
import random
import plotly.express as px
from sklearn.metrics import confusion_matrix

# 1. Page Configuration
st.set_page_config(
    page_title="Blue Team | Threat Monitor",
    page_icon="https://cdn-icons-png.flaticon.com/512/2092/2092663.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sidebar - Professional Project Details
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100) 
    st.title("Blue Team")
    st.markdown("**Cyber Anomaly Detection**")
    st.markdown("---")
    st.markdown("### Project Team:")
    st.markdown("- **Muhammad Ubaid** (Roll: BCS-48)")
    st.markdown("- **Shahid Khan** (Roll: BCS-90)")
    st.markdown("- **Saeed ur rahman** (Roll: BCS-59)")
    st.markdown("---")
    st.markdown("### Institution:")
    st.markdown("**BS Computer Science**\n\nAbdul Wali Khan University Mardan Timergara Campus")
    st.markdown("---")
    st.caption("System Status: ONLINE 🟢")

# 3. Load Models and Data 
@st.cache_resource
def load_assets():
    rf = joblib.load('cyber_ai_model.joblib')
    iso = joblib.load('iso_forest_model.joblib')
    data = pd.read_csv('KDDTrain_Cleaned.zip')
    return rf, iso, data

try:
    rf_model, iso_model, dataset = load_assets()
    X = dataset.drop('label', axis=1)
except Exception as e:
    st.error(f"Error loading models or data: {e}. Ensure training scripts were run.")
    st.stop()

# 4. Session State (Tracks intercepted packet and scan status)
if 'current_packet' not in st.session_state:
    st.session_state.current_packet = None
if 'packet_id' not in st.session_state:
    st.session_state.packet_id = None
if 'scan_status' not in st.session_state:
    st.session_state.scan_status = 'pending' # Options: 'pending', 'normal', 'attack'

# 5. App Header
st.title("🛡️ Network Threat Intelligence Dashboard")
st.markdown("Monitor, intercept, and analyze network traffic in real-time using dual-engine Machine Learning.")
st.markdown("---")

# 6. Main Dashboard Layout (Top Section)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📡 Live Packet Interceptor")
    st.write("Simulate capturing a live network packet from the traffic stream.")
    
    # Intercept Button grabs a new packet and resets the system to 'pending' (Blue)
    if st.button("Intercept Random Packet", type="primary", use_container_width=True):
        idx = random.randint(0, len(X) - 1)
        st.session_state.packet_id = idx
        st.session_state.current_packet = X.iloc[[idx]]
        st.session_state.scan_status = 'pending' 
        
    if st.session_state.current_packet is not None:
        st.success(f"Packet #{st.session_state.packet_id} Intercepted Successfully!")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Source Bytes", f"{st.session_state.current_packet['src_bytes'].values[0]:.2f}")
        metrics_col2.metric("Dest Bytes", f"{st.session_state.current_packet['dst_bytes'].values[0]:.2f}")
        metrics_col3.metric("Protocol", f"{st.session_state.current_packet['protocol_type'].values[0]:.2f}")

with col2:
    st.subheader("🧠 AI Analysis Engines")
    
    if st.session_state.current_packet is None:
        st.info("👈 Please intercept a network packet to begin analysis.")
    else:
        tab1, tab2 = st.tabs(["🎯 Supervised (Random Forest)", "🕵️ Unsupervised (Isolation Forest)"])
        
        with tab1:
            st.markdown("**Signature-Based Detection (99.95% Accuracy)**")
            if st.button("Run Deep Scan", key="btn_rf", use_container_width=True):
                prediction = rf_model.predict(st.session_state.current_packet)[0]
                if prediction == 1:
                    st.session_state.scan_status = 'attack' # Trigger Red Alert
                    st.error("🚨 CRITICAL ALERT: KNOWN CYBER ATTACK DETECTED")
                else:
                    st.session_state.scan_status = 'normal' # Trigger Green Safe
                    st.success("✅ TRAFFIC NORMAL: No known threat signatures found.")
                    
        with tab2:
            st.markdown("**Behavioral Anomaly Detection (Zero-Day Threats)**")
            if st.button("Run Behavioral Scan", key="btn_iso", use_container_width=True):
                prediction = iso_model.predict(st.session_state.current_packet)[0]
                if prediction == -1:
                    st.session_state.scan_status = 'attack' # Trigger Red Alert
                    st.warning("⚠️ WARNING: BEHAVIORAL ANOMALY DETECTED (Possible Zero-Day)")
                else:
                    st.session_state.scan_status = 'normal' # Trigger Green Safe
                    st.success("✅ TRAFFIC NORMAL: Behavior matches standard baselines.")

# 7. Live Analytics Section (Bottom Section)
st.markdown("---")
st.header("📊 Live System Analytics")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # --- SYNCHRONIZED RED ALERT LOGIC FOR CHART 1 ---
    if st.session_state.scan_status == 'attack':
        st.markdown("**🚨 Live Packet Signature (THREAT DETECTED) 🚨**")
        color_theme = 'Reds'
    elif st.session_state.scan_status == 'normal':
        st.markdown("**✅ Live Packet Signature (NORMAL)**")
        color_theme = 'Greens'
    else:
        st.markdown("**Live Packet Signature (Pending Scan...)**")
        color_theme = 'Blues'
        
    if st.session_state.current_packet is None:
        st.info("⏳ Waiting for packet interception...")
    else:
        packet_data = st.session_state.current_packet.iloc[0]
        features_to_show = ['src_bytes', 'dst_bytes', 'logged_in', 'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'diff_srv_rate']
        packet_subset = packet_data[features_to_show]
        
        df_live = pd.DataFrame({'Feature': packet_subset.index, 'Value': packet_subset.values})
        fig_live = px.bar(df_live, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale=color_theme)
        
        fig_live.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
        # Unique ID added here to prevent Duplicate Element Error
        st.plotly_chart(fig_live, use_container_width=True, key="live_signature_chart")

with chart_col2:
    # --- SYNCHRONIZED RED ALERT LOGIC FOR CHART 2 ---
    if st.session_state.scan_status == 'attack':
        st.markdown("**🚨 Live Accuracy Test (THREAT DETECTED) 🚨**")
        matrix_color = 'Reds'
    elif st.session_state.scan_status == 'normal':
        st.markdown("**✅ Live Accuracy Test (NORMAL)**")
        matrix_color = 'Greens'
    else:
        st.markdown("**Live Accuracy Test (10,000 Random Packets)**")
        matrix_color = 'Blues'

    sample_data = dataset.sample(n=10000)
    X_sample = sample_data.drop('label', axis=1)
    y_actual = sample_data['label']
    y_pred = rf_model.predict(X_sample)
    
    cm = confusion_matrix(y_actual, y_pred)
    
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=matrix_color,
                       labels=dict(x="AI Prediction", y="Actual Traffic"),
                       x=['Normal Traffic', 'Cyber Attack'], 
                       y=['Normal Traffic', 'Cyber Attack'])
    
    fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
    # Unique ID added here to prevent Duplicate Element Error
    st.plotly_chart(fig_cm, use_container_width=True, key="live_matrix_chart")


# 8. Intercepted Packet Data Viewer (Bottom Section)
st.markdown("---")
st.header("📂 Intercepted Packet Details")
st.write("View the complete raw data features of the currently intercepted network packet.")

if st.session_state.current_packet is None:
    st.info("⏳ Waiting for packet interception to display data...")
else:
    # Display only the single intercepted packet as a full scrollable table
    st.dataframe(st.session_state.current_packet, use_container_width=True)
