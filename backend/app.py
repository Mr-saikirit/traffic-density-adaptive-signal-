import sys
import os

# --- UPDATED FIX FOR backend/app.py location ---
# This tells Python to look in the parent folder (root) to find 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import tempfile
import time
import pandas as pd
import numpy as np
import base64
from datetime import datetime

# Now this will work even though app.py is inside /backend
from core.detection.vehicle_detector import VehicleDetector

# -------------------------------------------------
# 1. PAGE CONFIG & UI STYLING (RESTORED)
# -------------------------------------------------
st.set_page_config(page_title="Traffic AI Pro", layout="wide", page_icon="🚦")


def apply_global_glow(color_hex, is_emergency=False):
    """Restored past glow logic with an added pulse for emergency priority."""
    animation = "pulse 0.4s infinite" if is_emergency else "none"
    st.markdown(f"""
        <style>
        @keyframes pulse {{
            0% {{ background-color: rgba(255, 0, 0, 0.05); }}
            50% {{ background-color: rgba(255, 0, 0, 0.2); }}
            100% {{ background-color: rgba(255, 0, 0, 0.05); }}
        }}
        .stApp {{
            background-color: {color_hex}05; 
            border-top: 15px solid {color_hex};
            animation: {animation};
            transition: all 0.5s ease-in-out;
        }}
        [data-testid="stMetricValue"] {{ color: {color_hex}; }}
        </style>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# SAFE CSV PATH (CLOUD + LOCAL)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "analytics", "vehicles.csv")
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)


@st.cache_resource
def load_assets():
    return VehicleDetector()


detector = load_assets()

# -------------------- 2. LOGIN LOGIC (SAME AS PAST) --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Secure Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "password":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# -------------------------------------------------
# 3. SESSION STATE INIT
# -------------------------------------------------
if "run" not in st.session_state: st.session_state.run = False
# Essential for cumulative counting
if "ids" not in st.session_state: st.session_state.ids = set()
if "signal_color" not in st.session_state: st.session_state.signal_color = "#2ecc71"

# -------------------------------------------------
# 4. MAIN INTERFACE (RESTORED PAST UI)
# -------------------------------------------------
st.sidebar.header("🎛 System Controls")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

conf_val = st.sidebar.slider("AI Confidence", 0.1, 1.0, 0.35)
frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 2)
clearance_rate = st.sidebar.number_input("Sec/Vehicle", value=2.5)

uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

st.title("🚦 Traffic Density Analysis & Adaptive Signal Control")
apply_global_glow(st.session_state.signal_color)

tab1, tab2, tab3 = st.tabs(["🚥 Live Traffic", "📊 Analytics", "🧠 System Info"])

# ======================================================
# TAB 1: LIVE TRAFFIC
# ======================================================
with tab1:
    if not uploaded_file:
        st.info("Please upload a video file in the sidebar to begin.")
    else:
        col_video, col_metrics = st.columns([2, 1])

        with col_video:
            video_placeholder = st.empty()
            c1, c2 = st.columns(2)
            if c1.button("▶ Start Analysis", use_container_width=True):
                st.session_state.run = True
                st.session_state.ids = set()  # Clear previous run counts

            if c2.button("⏹ Stop & Save", use_container_width=True):
                st.session_state.run = False
                total_unique = len(st.session_state.ids)

                log_entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Total_Vehicles": total_unique,
                    "Signal_Time_Sec": round(total_unique * clearance_rate, 2)
                }
                pd.DataFrame([log_entry]).to_csv(
                    CSV_FILE,
                    mode='a',
                    index=False,
                    header=not os.path.exists(CSV_FILE)
                )
                st.success(f"✅ Data Logged! Total Unique Vehicles: {total_unique}")

        with col_metrics:
            st.subheader("📌 Live Metrics")
            dens_m = st.empty()
            total_m = st.empty()
            status_box = st.empty()

        if st.session_state.run:
            # ---------------- SAFE TEMP VIDEO HANDLING ----------------
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            cap = cv2.VideoCapture(video_path)
            frame_counter = 0

            while cap.isOpened() and st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_counter += 1

                if frame_counter % frame_skip == 0:
                    detections, is_emergency = detector.process_frame(frame, conf_val)

                    norm_count = 0
                    for d in detections:
                        if d["type"] == "normal":
                            norm_count += 1
                            st.session_state.ids.add(d["id"])

                    if is_emergency:
                        st.session_state.signal_color = "#FF0000"
                        status_box.error("🚨 EMERGENCY VEHICLE DETECTED")
                    else:
                        if norm_count > 15:
                            st.session_state.signal_color = "#e74c3c"
                        elif norm_count > 5:
                            st.session_state.signal_color = "#f1c40f"
                        else:
                            st.session_state.signal_color = "#2ecc71"

                        if norm_count > 15:
                            status_box.error("🔴 RED – High Density")
                        elif norm_count > 5:
                            status_box.warning("🟡 YELLOW – Medium Density")
                        else:
                            status_box.success("🟢 GREEN – Low Density")

                    apply_global_glow(st.session_state.signal_color, is_emergency)

                    dens_m.metric("🚗 Frame Density", norm_count)
                    total_m.metric("📈 Cumulative Total", len(st.session_state.ids))
                    st.session_state.last_detections = detections
                else:
                    detections = st.session_state.get("last_detections", [])

                display_frame = cv2.resize(frame, (854, 480))
                scale_x = 854 / frame.shape[1]
                scale_y = 480 / frame.shape[0]

                for d in detections:
                    x1, y1, x2, y2 = d["box"]
                    color = (0, 0, 255) if d["type"] == "emergency" else (0, 255, 0)
                    cv2.rectangle(
                        display_frame,
                        (int(x1 * scale_x), int(y1 * scale_y)),
                        (int(x2 * scale_x), int(y2 * scale_y)),
                        color,
                        2
                    )

                
                video_placeholder.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    use_column_width=True
)

                
            
                time.sleep(0.01)

            cap.release()

# ======================================================
# TAB 2: ANALYTICS (FIXED FOR PARSERERROR)
# ======================================================
with tab2:
    st.subheader("📊 Historical Traffic Comparison")
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
            if not df.empty:
                df['Execution'] = df['Date'] + " " + df['Time']
                st.write("### Total Vehicles Per Execution Cycle")
                st.bar_chart(df.set_index("Execution")["Total_Vehicles"])

                st.divider()
                st.write("### 📜 Raw Execution History")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No data in CSV yet.")
        except Exception as e:
            st.error(f"Error reading history: {e}")
    else:
        st.info("No data available. Complete an analysis run to see charts.")

# ======================================================
# TAB 3: SYSTEM INFO
# ======================================================
with tab3:
    st.markdown("""
    **Accuracy & Logic:**
    * **ID Persistence:** Uses ByteTrack IDs to ensure a vehicle is only counted once for the entire video run.
    * **Dynamic Scaling:** Bounding boxes are scaled from original video resolution to UI display resolution.
    * **Error Resiliency:** The CSV loader skips corrupt lines to ensure the Analytics tab always works.
    """)
