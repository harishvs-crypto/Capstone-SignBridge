import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Import backend modules
from backend.hand_detector import HandDetector
from backend.model_utils import SignLanguageClassifier
from backend.speech_generator import SpeechGenerator
from backend.inference_pipeline import (
    PredictionStabilizer,
    build_debug_overlay,
    build_model_input_preview,
    prepare_roi_for_inference,
    CONFIDENCE_THRESHOLD,
)

# Page config
st.set_page_config(
    page_title="SignBridge - ISL to Speech",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False
if 'last_predicted' not in st.session_state:
    st.session_state.last_predicted = None
if 'stable_count' not in st.session_state:
    st.session_state.stable_count = 0
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = False
if 'last_appended' not in st.session_state:
    st.session_state.last_appended = None

# Load classifier and speech engine
@st.cache_resource
def get_classifier():
    return SignLanguageClassifier()

@st.cache_resource
def get_detector():
    return HandDetector()

@st.cache_resource
def get_speech_generator():
    return SpeechGenerator()

classifier = get_classifier()
detector = get_detector()
speech_gen = get_speech_generator()

# Custom CSS for UI styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global Overrides */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background & Page Wrapper overrides */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #15172b 0%, #07080e 100%) !important;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0c0d17 !important;
        border-right: 1px solid rgba(127, 0, 255, 0.15) !important;
    }
    
    /* Hide default Streamlit decoration line at top and default footer */
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #ff007f, #7f00ff, #00f0ff) !important;
        height: 4px !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header Container styling */
    .header-container {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, rgba(30, 30, 54, 0.6) 0%, rgba(12, 13, 23, 0.8) 100%);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(127, 0, 255, 0.2);
        box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 240, 255, 0.6), rgba(255, 0, 127, 0.6), transparent);
    }
    
    .project-badge {
        display: inline-block;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #00f0ff;
        border: 1px solid rgba(0, 240, 255, 0.4);
        background: rgba(0, 240, 255, 0.05);
        border-radius: 30px;
        letter-spacing: 2px;
        margin-bottom: 1.2rem;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.1);
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f0ff, #7f00ff, #ff007f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        color: #a0a5c0;
        font-size: 1.25rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Card Glassmorphic styling */
    .glass-card {
        background: rgba(20, 22, 43, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(127, 0, 255, 0.35);
        box-shadow: 0 15px 35px 0 rgba(127, 0, 255, 0.15), 0 5px 15px 0 rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Glowing cyan/purple accent line on card titles */
    .card-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 18px;
        background: linear-gradient(180deg, #00f0ff, #7f00ff);
        border-radius: 4px;
    }
    
    /* Display parameters */
    .metric-value {
        font-size: 5rem;
        font-weight: 800;
        color: #ff007f;
        text-align: center;
        margin: 15px 0;
        text-shadow: 0 0 30px rgba(255, 0, 127, 0.5);
        letter-spacing: -2px;
        animation: pulseGlow 2s infinite ease-in-out;
    }
    
    @keyframes pulseGlow {
        0% { text-shadow: 0 0 20px rgba(255, 0, 127, 0.4); }
        50% { text-shadow: 0 0 40px rgba(255, 0, 127, 0.7), 0 0 10px rgba(255, 0, 127, 0.3); }
        100% { text-shadow: 0 0 20px rgba(255, 0, 127, 0.4); }
    }
    
    .metric-confidence {
        font-size: 1.3rem;
        color: #00f0ff;
        text-align: center;
        font-weight: 700;
        letter-spacing: 0.5px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Buffer Text Output Display */
    .text-buffer-box {
        background-color: rgba(10, 11, 20, 0.85);
        border: 2px dashed rgba(127, 0, 255, 0.5);
        border-radius: 16px;
        padding: 1.8rem;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #ffffff;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 4px 15px rgba(0, 0, 0, 0.6);
        transition: all 0.3s ease;
    }
    
    .text-buffer-box:hover {
        border-color: #00f0ff;
        box-shadow: inset 0 4px 15px rgba(0, 0, 0, 0.6), 0 0 15px rgba(0, 240, 255, 0.2);
    }
    
    .placeholder-text {
        color: #464a66;
        font-style: italic;
        font-size: 1.3rem;
        font-weight: 400;
        letter-spacing: 0;
    }
    
    /* Universal Button Customization */
    div.stButton > button {
        width: 100% !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e0e2ed !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    }
    
    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: #7f00ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(127, 0, 255, 0.35) !important;
        transform: translateY(-2px);
    }
    
    /* Primary (Speak) Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7f00ff 0%, #ff007f 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(127, 0, 255, 0.35) !important;
        font-weight: 700 !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 25px rgba(255, 0, 127, 0.5) !important;
        transform: translateY(-2px) scale(1.02);
    }
    
    /* Streamlit Input Styling: Sliders & Controls */
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: #7f00ff !important;
    }
    
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #00f0ff !important;
        border: 2px solid #7f00ff !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.8) !important;
    }
    
    div[data-testid="stCheckbox"] label span {
        color: #d1d4e6 !important;
        font-weight: 500;
    }
    
    div[data-testid="stRadio"] label {
        color: #d1d4e6 !important;
        font-weight: 500;
    }
    
    /* Image and Video Output Previews */
    [data-testid="stImage"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stImage"]:hover {
        border-color: rgba(0, 240, 255, 0.4) !important;
        box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15) !important;
    }
    
    /* Progress Bars styling */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #7f00ff, #00f0ff) !important;
        border-radius: 10px !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        color: #585d7a;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="header-container">
    <span class="project-badge">NVIDIA CAPSTONE PROJECT</span>
    <div class="header-title">SignBridge</div>
    <div class="header-subtitle">Real-Time Indian Sign Language to Speech Translation</div>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown('<div class="card-title">System Settings</div>', unsafe_allow_html=True)
    
    # Input Mode
    mode = st.radio("Choose Input Mode", ["Live Webcam", "Simulator Demo"], index=0)
    st.session_state.simulation_mode = (mode == "Simulator Demo")
    
    st.markdown("---")
    
    # Detection threshold (80% minimum to avoid wrong letters)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    
    # Stability buffer limit (number of frames before confirmation)
    stability_limit = st.slider("Auto-Append Hold Frames", 5, 30, 15, 1)
    
    # Auto-append Toggle
    auto_append = st.checkbox("Auto-Append Letters", value=True, help="Automatically append letters when held steady.")
    
    # Developer mode toggle
    dev_mode = st.checkbox("🔧 Developer Debug Mode", value=False, help="Show prediction details, probabilities, FPS, and model inputs.")

    st.markdown("---")
    
    # Skin color / Brightness calibrator
    st.markdown('<div class="card-title" style="font-size: 1.1rem; margin-top: 0.5rem;">🎛️ Segmentation Calibrator</div>', unsafe_allow_html=True)
    st.caption("Adjust these settings if your hand is not being tracked properly under your room lighting.")
    seg_mode = st.selectbox("Tracking Mode", ["Skin Color", "Brightness/Grayscale"], index=0)
    
    if seg_mode == "Skin Color":
        cr_min = st.slider("Cr (Redness) Min", 80, 150, 130)
        cr_max = st.slider("Cr (Redness) Max", 150, 220, 180)
        cb_min = st.slider("Cb (Blueness) Min", 60, 120, 77)
        cb_max = st.slider("Cb (Blueness) Max", 120, 180, 127)
        bright_thresh = 40
    else:
        bright_thresh = st.slider("Brightness Threshold", 10, 150, 40)
        cr_min, cr_max, cb_min, cb_max = 130, 180, 77, 127
        
    lower_skin_arr = np.array([0, cr_min, cb_min], dtype=np.uint8)
    upper_skin_arr = np.array([255, cr_max, cb_max], dtype=np.uint8)

    st.markdown("---")
    
    # Information Card
    st.markdown("""
    **Quick Guide:**
    1. Position your hand inside the frame.
    2. MediaPipe detects your hand and extracts a bounding box.
    3. The CNN classifies the gesture (A-Z).
    4. Auto-append adds the letter to the buffer once stable.
    5. Click **Speak** to hear your text!
    """)
    
    # Display model state
    if classifier.model is None:
        st.error("Model is not loaded. Please train using `train.py`.")
    else:
        st.success("CNN Classification Model: LOADED")

# Main Page Layout (Two Columns)
col1, col2 = st.columns([3, 2])

# Control logic inside col2
with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Translation Control & Display</div>', unsafe_allow_html=True)
    
    # Dynamic display of prediction
    pred_placeholder = st.empty()
    conf_placeholder = st.empty()
    stable_bar_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Floating Dev Dashboard Placeholder
    dev_card_placeholder = st.empty()
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Generated Text</div>', unsafe_allow_html=True)
    
    # Display current text buffer
    buffer_placeholder = st.empty()
    
    # Buffer buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🔊 Speak", type="primary"):
            speech_gen.speak_text(st.session_state.generated_text)
            
    with btn_col2:
        if st.button("⌫ Delete"):
            if len(st.session_state.generated_text) > 0:
                st.session_state.generated_text = st.session_state.generated_text[:-1]
                
    with btn_col3:
        if st.button("🗑️ Clear"):
            st.session_state.generated_text = ""
            st.session_state.last_appended = None
            
    # Add a Space button
    if st.button("␣ Space"):
        st.session_state.generated_text += " "
        st.session_state.last_appended = " "
        
    # Option to manually add the current letter
    if st.button("➕ Add Prediction"):
        if st.session_state.last_predicted and st.session_state.last_predicted != "No Hand":
            st.session_state.generated_text += st.session_state.last_predicted
            st.session_state.last_appended = st.session_state.last_predicted

    st.markdown('</div>', unsafe_allow_html=True)

# Helper function to render prediction states
def update_prediction_ui(letter, confidence, progress_val=0):
    if letter == "No Hand" or letter == "Unknown":
        pred_placeholder.markdown(f'<div class="metric-value" style="color: #4a4d6b; font-size: 2.5rem;">{letter.upper()}</div>', unsafe_allow_html=True)
        conf_placeholder.markdown(f'<div class="metric-confidence" style="color: #4a4d6b;">Confidence: N/A</div>', unsafe_allow_html=True)
        stable_bar_placeholder.empty()
    else:
        color = "#00f0ff" if confidence >= confidence_threshold else "#ff007f"
        display_letter = "Space" if letter == " " or letter == "{" else letter
        pred_placeholder.markdown(f'<div class="metric-value" style="color: {color};">{display_letter}</div>', unsafe_allow_html=True)
        conf_placeholder.markdown(f'<div class="metric-confidence" style="color: {color};">Confidence: {confidence*100:.1f}%</div>', unsafe_allow_html=True)
        if auto_append:
            stable_bar_placeholder.progress(progress_val)
        else:
            stable_bar_placeholder.empty()

    # Update Text Buffer Box
    if st.session_state.generated_text.strip() == "":
        buffer_placeholder.markdown('<div class="text-buffer-box"><span class="placeholder-text">Waiting for signs...</span></div>', unsafe_allow_html=True)
    else:
        buffer_placeholder.markdown(f'<div class="text-buffer-box">{st.session_state.generated_text}</div>', unsafe_allow_html=True)

# Webcam Column (col1)
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Video Feed</div>', unsafe_allow_html=True)
    
    # Controls inside the column
    run_btn_col, stop_btn_col = st.columns(2)
    with run_btn_col:
        start_webcam = st.button("📹 Start Capture", use_container_width=True)
    with stop_btn_col:
        stop_webcam = st.button("🛑 Stop Capture", use_container_width=True)
        
    if start_webcam:
        st.session_state.run_webcam = True
    if stop_webcam:
        st.session_state.run_webcam = False
        
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Run Loop
if st.session_state.run_webcam:
    if st.session_state.simulation_mode:
        # SIMULATION MODE (Generates moving letters & overlays, perfect for environments without camera)
        classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [' ']
        sim_index = 0
        tick = 0
        
        while st.session_state.run_webcam:
            # Construct a synthetic hand frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add simple background gradient/grid
            cv2.rectangle(frame, (10, 10), (630, 470), (20, 20, 30), -1)
            for i in range(0, 640, 40):
                cv2.line(frame, (i, 0), (i, 480), (30, 30, 45), 1)
            for i in range(0, 480, 40):
                cv2.line(frame, (0, i), (640, i), (30, 30, 45), 1)
                
            # Current simulated letter
            sim_letter = classes[sim_index]
            
            # Simulate a "hand" ROI in the frame
            hand_x, hand_y = 220, 140
            hand_w, hand_h = 200, 200
            
            # Draw simulation bounding box
            cv2.rectangle(frame, (hand_x, hand_y), (hand_x + hand_w, hand_y + hand_h), (0, 255, 0), 2)
            cv2.putText(frame, "SIMULATED HAND ROI", (hand_x, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw simulated hand circle and text
            cv2.circle(frame, (hand_x + 100, hand_y + 100), 60, (220, 180, 160), -1)
            cv2.putText(frame, sim_letter, (hand_x + 80, hand_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            
            # Prediction values
            pred_letter = sim_letter
            pred_conf = 0.85 + 0.1 * np.sin(tick * 0.1) # Confidence oscillates slightly
            
            # Increment stability
            if st.session_state.last_predicted == pred_letter:
                st.session_state.stable_count += 1
            else:
                st.session_state.stable_count = 0
                st.session_state.last_predicted = pred_letter
                
            progress = min(1.0, st.session_state.stable_count / stability_limit)
            
            if auto_append and st.session_state.stable_count >= stability_limit:
                if pred_letter != st.session_state.last_appended:
                    st.session_state.generated_text += pred_letter
                    st.session_state.last_appended = pred_letter
                st.session_state.stable_count = 0
                
            # Render frame
            cv2.putText(frame, f"MODE: SIMULATOR (Letter: {sim_letter})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)
            cv2.putText(frame, "Switch sidebar to Live Webcam for physical camera", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB")
            
            # Update UI
            update_prediction_ui(pred_letter, pred_conf, progress)
            
            # Simulated Dev mode support
            if dev_mode:
                with dev_card_placeholder.container():
                    st.markdown('<div class="glass-card" style="border-color: #ff007f;">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title" style="color: #ff007f;">🔧 Developer Debug Dashboard</div>', unsafe_allow_html=True)
                    st.metric("FPS (Simulation)", "30.0 Hz")
                    st.write(f"**Stability Count:** {st.session_state.stable_count} / {stability_limit}")
                    st.write(f"**Last Appended:** '{st.session_state.last_appended}'")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                dev_card_placeholder.empty()
                
            time.sleep(0.08)
            tick += 1
            
            # Change simulated letter every ~40 ticks (~3 seconds)
            if tick % 40 == 0:
                sim_index = (sim_index + 1) % len(classes)
                
    else:
        # LIVE WEBCAM MODE
        cap = cv2.VideoCapture(0)
        
        # Configure lower latency/resolution for webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Could not open webcam. If you do not have a physical webcam, please switch to 'Simulator Demo' in the sidebar.")
            st.session_state.run_webcam = False
        else:
            stabilizer = PredictionStabilizer()
            
            # FPS tracking variables
            fps_start_time = time.time()
            fps_counter = 0
            fps = 0.0
            
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam.")
                    break
                
                # FPS calculation
                fps_counter += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time >= 1.0:
                    fps = fps_counter / elapsed_time
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Flip frame horizontally for natural mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process hand landmark detection
                landmarks_results = detector.find_hand_landmarks(
                    frame, 
                    mode=seg_mode, 
                    threshold_val=bright_thresh, 
                    lower_skin=lower_skin_arr, 
                    upper_skin=upper_skin_arr
                )
                
                # Extract ROI
                cropped_roi, bbox, annotated_frame = detector.get_hand_roi(
                    frame, 
                    landmarks_results, 
                    mode=seg_mode, 
                    threshold_val=bright_thresh, 
                    lower_skin=lower_skin_arr, 
                    upper_skin=upper_skin_arr
                )
                
                pred_letter = "No Hand"
                pred_conf = 0.0
                top_3 = []
                rotated_roi = None
                result = {}
                
                if cropped_roi is not None:
                    # Run classifier through fixed inference pipeline (rotate + filter + stabilize)
                    result = stabilizer.predict_from_roi(classifier, cropped_roi, apply_rotation=True)
                    pred_letter = result["letter"]
                    pred_conf = result["confidence"]
                    top_3 = result.get("top_3", [])
                    rotated_roi = result.get("rotated_roi")
                    
                    if pred_letter and pred_letter not in ("Mock (Model Not Loaded)", "Unknown"):
                        # Process stability auto-append
                        if pred_conf >= confidence_threshold:
                            if st.session_state.last_predicted == pred_letter:
                                st.session_state.stable_count += 1
                            else:
                                st.session_state.stable_count = 0
                                st.session_state.last_predicted = pred_letter
                        else:
                            st.session_state.stable_count = 0
                    elif pred_letter == "Unknown":
                        st.session_state.stable_count = 0
                        st.session_state.last_predicted = "Unknown"
                    else:
                        st.session_state.stable_count = 0
                else:
                    stabilizer.reset()
                    st.session_state.stable_count = 0
                    st.session_state.last_predicted = "No Hand"
                    st.session_state.last_appended = None # Reset duplicate protection
                
                # Auto-append stable character to buffer
                progress = min(1.0, st.session_state.stable_count / stability_limit)
                if auto_append and pred_letter != "No Hand" and pred_letter != "Unknown" and st.session_state.stable_count >= stability_limit:
                    if pred_conf >= confidence_threshold:
                        # Prevent duplicate letters being appended repeatedly
                        if pred_letter != st.session_state.last_appended:
                            st.session_state.generated_text += pred_letter
                            st.session_state.last_appended = pred_letter
                        st.session_state.stable_count = 0
                
                # Render video frame to Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB")
                
                # Update output widgets in col2
                update_prediction_ui(pred_letter, pred_conf, progress)
                
                # Update Developer Debug Dashboard if active
                if dev_mode:
                    with dev_card_placeholder.container():
                        st.markdown('<div class="glass-card" style="border-color: #ff007f; padding: 1rem;">', unsafe_allow_html=True)
                        st.markdown('<div class="card-title" style="color: #ff007f; font-size: 1.1rem; margin-bottom: 0.5rem;">🔧 Developer Debug Dashboard</div>', unsafe_allow_html=True)
                        
                        # Core performance metrics
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("FPS (Processing Speed)", f"{fps:.1f} Hz")
                            st.write(f"**Hand Status:** {'Detected' if cropped_roi is not None else 'Not Found'}")
                        with metric_col2:
                            st.write(f"**Bounding Box:** {bbox if bbox else 'N/A'}")
                            st.write(f"**Stability:** {st.session_state.stable_count} / {stability_limit}")
                        
                        # Preprocessed image preview (pipeline stages)
                        if cropped_roi is not None:
                            st.markdown("---")
                            st.write("**Pipeline Debug (Webcam → Crop → Rotate → CNN Input):**")
                            dbg_col1, dbg_col2, dbg_col3 = st.columns(3)
                            with dbg_col1:
                                st.image(cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB), caption="Cropped ROI", width=120)
                            with dbg_col2:
                                if rotated_roi is not None:
                                    st.image(cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2RGB), caption="Rotated ROI (90° CW)", width=120)
                            with dbg_col3:
                                model_preview = build_model_input_preview(classifier, rotated_roi or cropped_roi)
                                st.image(cv2.cvtColor(model_preview, cv2.COLOR_BGR2RGB), caption="CNN Input (64×64)", width=120)

                            overlay = build_debug_overlay(
                                frame=annotated_frame,
                                bbox=bbox,
                                cropped_roi=cropped_roi,
                                rotated_roi=rotated_roi,
                                model_input=model_preview,
                                letter=pred_letter,
                                confidence=pred_conf,
                                raw_letter=result.get("raw_letter", ""),
                                raw_confidence=result.get("raw_confidence", 0.0),
                            )
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Full Debug Overlay", use_container_width=True)
                        
                        # Top-3 Predictions display
                        if len(top_3) > 0:
                            st.markdown("---")
                            st.write("**Top-3 Probabilities:**")
                            for label, val in top_3:
                                st.write(f"Class **{label}**: {val*100:.1f}%")
                                st.progress(val)
                                
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    dev_card_placeholder.empty()
                
                # Small yield for Streamlit GUI responsiveness
                time.sleep(0.01)
                
            cap.release()
            
else:
    # Webcam stopped, show a static placeholder
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = [15, 17, 26] # dark background
    cv2.putText(img, "Camera Stopped", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 120), 2)
    video_placeholder.image(img, channels="RGB")
    update_prediction_ui("No Hand", 0.0, 0.0)
    dev_card_placeholder.empty()

# Footer
st.markdown("""
<div class="footer">
    SignBridge Capstone Project V2 • Powered by MediaPipe, OpenCV & TensorFlow
</div>
""", unsafe_allow_html=True)
