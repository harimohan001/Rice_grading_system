import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Rice Grading System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS + ANIMATIONS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap');

/* ═══════════════════════════════════
   ROOT VARIABLES
═══════════════════════════════════ */
:root {
    --sky:       #e0f4ff;
    --sky2:      #b8e4f9;
    --sky3:      #7cc8f0;
    --skyDeep:   #3a9fd6;
    --teal:      #1a7fa8;
    --tealDark:  #0d5a7a;
    --gold:      #e8a020;
    --gold2:     #f5c842;
    --white:     #ffffff;
    --text:      #0d3a52;
    --textSub:   #4a7a96;
    --card:      rgba(255,255,255,0.72);
    --cardBdr:   rgba(58,159,214,0.22);
    --shadow:    rgba(26,127,168,0.14);
}

/* ═══════════════════════════════════
   BASE + BACKGROUND
═══════════════════════════════════ */
html, body, .stApp {
    background: linear-gradient(160deg, #c8eeff 0%, #e8f7ff 40%, #d0f0e8 100%) !important;
    font-family: 'Nunito', sans-serif;
    color: var(--text);
    min-height: 100vh;
}

/* animated sky gradient blobs */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 10% 15%, rgba(124,200,240,0.45) 0%, transparent 65%),
        radial-gradient(ellipse 55% 60% at 90% 80%, rgba(180,230,200,0.35) 0%, transparent 65%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(255,255,255,0.2) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: skyBlobs 12s ease-in-out infinite alternate;
}

@keyframes skyBlobs {
    0%   { opacity: 0.7; transform: scale(1) translateX(0); }
    50%  { opacity: 1;   transform: scale(1.03) translateX(8px); }
    100% { opacity: 0.85; transform: scale(0.98) translateX(-5px); }
}

/* ═══════════════════════════════════
   RICE STALK DECORATIONS
═══════════════════════════════════ */
.rice-left, .rice-right {
    position: fixed;
    top: 0;
    width: 140px;
    height: 100vh;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}
.rice-left  { left: 0; }
.rice-right { right: 0; transform: scaleX(-1); }

.rice-stalk {
    position: absolute;
    bottom: 0;
    font-size: 2.2rem;
    animation: stalksway 3.5s ease-in-out infinite alternate;
    transform-origin: bottom center;
    filter: drop-shadow(0 2px 8px rgba(26,127,168,0.18));
    display: flex;
    align-items: flex-end;
}
.rice-stalk:nth-child(1) { left:10px;  animation-delay:0s;    font-size:2.8rem; height:320px; }
.rice-stalk:nth-child(2) { left:45px;  animation-delay:0.6s;  font-size:2.2rem; height:260px; }
.rice-stalk:nth-child(3) { left:80px;  animation-delay:1.1s;  font-size:3rem;   height:380px; }
.rice-stalk:nth-child(4) { left:115px; animation-delay:0.3s;  font-size:2rem;   height:220px; }

@keyframes stalksway {
    0%   { transform: rotate(-6deg); }
    100% { transform: rotate(6deg); }
}

/* floating grain particles */
.grain-float {
    position: fixed;
    font-size: 1.1rem;
    opacity: 0;
    pointer-events: none;
    z-index: 1;
    animation: grainFloat linear infinite;
}
@keyframes grainFloat {
    0%   { opacity: 0;    transform: translateY(0)    rotate(0deg); }
    10%  { opacity: 0.55; }
    90%  { opacity: 0.4; }
    100% { opacity: 0;    transform: translateY(-95vh) rotate(360deg); }
}

/* ═══════════════════════════════════
   HERO HEADER
═══════════════════════════════════ */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 2.4rem 1rem 0.8rem;
    z-index: 2;
}
.hero-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--teal);
    border: 1.5px solid rgba(26,127,168,0.28);
    border-radius: 50px;
    padding: 5px 20px;
    margin-bottom: 16px;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(8px);
    animation: fadeUp 0.6s ease both;
    box-shadow: 0 2px 12px rgba(26,127,168,0.10);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 5.5vw, 4.8rem);
    font-weight: 900;
    line-height: 1.1;
    color: var(--tealDark);
    margin: 0 0 10px;
    animation: fadeUp 0.65s 0.1s ease both;
    text-shadow: 0 2px 24px rgba(26,127,168,0.12);
}
.hero-title span {
    background: linear-gradient(135deg, var(--skyDeep) 0%, var(--teal) 40%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--textSub);
    font-weight: 400;
    letter-spacing: 0.03em;
    animation: fadeUp 0.65s 0.2s ease both;
}
.hero-line {
    width: 90px;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--skyDeep), var(--gold), transparent);
    margin: 16px auto 0;
    border-radius: 2px;
    animation: expandLine 1s 0.4s ease both;
}
@keyframes expandLine {
    from { transform: scaleX(0); opacity: 0; }
    to   { transform: scaleX(1); opacity: 1; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════
   LOADING ANIMATION
═══════════════════════════════════ */
.loading-scene {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 18px;
    padding: 40px 20px;
    background: rgba(255,255,255,0.65);
    border-radius: 24px;
    border: 1.5px solid var(--cardBdr);
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px var(--shadow);
    animation: fadeUp 0.4s ease both;
    margin: 10px 0;
}
.loading-person {
    font-size: 3.8rem;
    animation: personInspect 1.2s ease-in-out infinite alternate;
    filter: drop-shadow(0 4px 12px rgba(26,127,168,0.25));
}
@keyframes personInspect {
    0%   { transform: translateY(0) rotate(-5deg) scale(1); }
    100% { transform: translateY(-10px) rotate(5deg) scale(1.05); }
}
.conveyor-wrap {
    position: relative;
    width: 300px;
    height: 46px;
}
.conveyor-track {
    position: absolute;
    bottom: 0;
    width: 100%;
    height: 14px;
    background: repeating-linear-gradient(
        90deg,
        var(--sky2) 0px, var(--sky2) 18px,
        var(--skyDeep) 18px, var(--skyDeep) 20px
    );
    border-radius: 7px;
    animation: conveyorMove 0.5s linear infinite;
    box-shadow: 0 3px 10px rgba(26,127,168,0.22);
}
@keyframes conveyorMove {
    from { background-position: 0 0; }
    to   { background-position: 20px 0; }
}
.grain-belt {
    position: absolute;
    top: 4px;
    font-size: 1.4rem;
    animation: grainRide 2.2s linear infinite;
}
.grain-belt:nth-child(2) { animation-delay:0.73s; }
.grain-belt:nth-child(3) { animation-delay:1.46s; }
@keyframes grainRide {
    from { left: -40px; opacity: 0; }
    8%   { opacity: 1; }
    92%  { opacity: 1; }
    to   { left: 320px; opacity: 0; }
}
.loading-bars {
    display: flex;
    gap: 6px;
    align-items: flex-end;
    height: 40px;
}
.loading-bar {
    width: 9px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--skyDeep), var(--gold));
    animation: barBounce 0.85s ease-in-out infinite alternate;
    box-shadow: 0 2px 8px rgba(58,159,214,0.3);
}
.loading-bar:nth-child(1) { animation-delay:0s;    height:18px; }
.loading-bar:nth-child(2) { animation-delay:0.12s; height:28px; }
.loading-bar:nth-child(3) { animation-delay:0.24s; height:38px; }
.loading-bar:nth-child(4) { animation-delay:0.36s; height:26px; }
.loading-bar:nth-child(5) { animation-delay:0.48s; height:16px; }
.loading-bar:nth-child(6) { animation-delay:0.60s; height:30px; }
@keyframes barBounce {
    from { transform: scaleY(0.35); opacity: 0.45; }
    to   { transform: scaleY(1.0);  opacity: 1.0; }
}
.loading-label {
    font-size: 0.95rem;
    font-weight: 800;
    color: var(--teal);
    letter-spacing: 0.08em;
    animation: labelPulse 1.4s ease-in-out infinite;
}
@keyframes labelPulse {
    0%,100% { opacity: 0.55; }
    50%     { opacity: 1.0; }
}
.loading-dots::after {
    content: '';
    animation: dots 1.8s steps(4,end) infinite;
}
@keyframes dots {
    0%  { content: '.'; }
    33% { content: '..'; }
    66% { content: '...'; }
    99% { content: ''; }
}

/* ═══════════════════════════════════
   METRIC CARDS
═══════════════════════════════════ */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.92) !important;
    border: 1.5px solid var(--cardBdr) !important;
    border-radius: 18px !important;
    padding: 18px 14px !important;
    backdrop-filter: blur(14px);
    box-shadow: 0 4px 20px var(--shadow), inset 0 1px 0 rgba(255,255,255,0.95);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: cardIn 0.5s ease both;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(26,127,168,0.22);
}
/* label (e.g. "TOTAL GRAINS") */
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] span {
    color: #1a7fa8 !important;
    font-size: 0.7rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    opacity: 1 !important;
}
/* value (e.g. "19") */
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] *  {
    color: #000000 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
/* delta text if any */
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #1a7fa8 !important;
}
@keyframes cardIn {
    from { opacity: 0; transform: translateY(18px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
}

/* ═══════════════════════════════════
   SIDEBAR
═══════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d5a7a 0%, #1a7fa8 65%, #2dafd8 100%) !important;
    border-right: 1.5px solid rgba(255,255,255,0.18);
    box-shadow: 4px 0 30px rgba(13,90,122,0.28);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

/* ═══════════════════════════════════
   BUTTONS
═══════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--teal), var(--skyDeep)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(26,127,168,0.28) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(26,127,168,0.4) !important;
}

/* ═══════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════ */
.stFileUploader {
    border: 2.5px dashed rgba(58,159,214,0.45) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    border-color: var(--skyDeep) !important;
    background: rgba(255,255,255,0.65) !important;
    box-shadow: 0 4px 24px rgba(26,127,168,0.15);
}

/* ═══════════════════════════════════
   IMAGES
═══════════════════════════════════ */
.stImage img {
    border-radius: 16px;
    box-shadow: 0 6px 28px rgba(26,127,168,0.18);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stImage img:hover {
    transform: scale(1.015);
    box-shadow: 0 12px 40px rgba(26,127,168,0.28);
}

/* ═══════════════════════════════════
   ALERTS
═══════════════════════════════════ */
.stAlert {
    border-radius: 14px !important;
    backdrop-filter: blur(8px);
    animation: slideAlert 0.4s ease;
}
@keyframes slideAlert {
    from { opacity: 0; transform: translateX(-14px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ═══════════════════════════════════
   STATUS BADGES
═══════════════════════════════════ */
.status-pass, .status-fail, .status-warn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 24px;
    border-radius: 50px;
    font-size: 0.95rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    animation: statusPop 0.45s cubic-bezier(.34,1.56,.64,1) both;
}
.status-pass {
    background: rgba(40,190,110,0.14);
    border: 1.5px solid rgba(40,190,110,0.4);
    color: #1a8a50;
    animation: statusPop 0.45s cubic-bezier(.34,1.56,.64,1) both, glowGreen 2.4s ease-in-out infinite;
}
.status-fail {
    background: rgba(220,60,60,0.12);
    border: 1.5px solid rgba(220,60,60,0.38);
    color: #c03030;
    animation: statusPop 0.45s cubic-bezier(.34,1.56,.64,1) both, glowRed 1.8s ease-in-out infinite;
}
.status-warn {
    background: rgba(232,160,32,0.12);
    border: 1.5px solid rgba(232,160,32,0.4);
    color: #a05800;
}
@keyframes statusPop {
    from { opacity: 0; transform: scale(0.78); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes glowGreen {
    0%,100% { box-shadow: 0 0 0 0   rgba(40,190,110,0.22); }
    50%     { box-shadow: 0 0 0 10px rgba(40,190,110,0); }
}
@keyframes glowRed {
    0%,100% { box-shadow: 0 0 0 0   rgba(220,60,60,0.22); }
    50%     { box-shadow: 0 0 0 10px rgba(220,60,60,0); }
}

/* ═══════════════════════════════════
   CHART + DATAFRAME
═══════════════════════════════════ */
[data-testid="stVegaLiteChart"] {
    background: var(--card) !important;
    border-radius: 16px !important;
    border: 1px solid var(--cardBdr) !important;
    padding: 14px !important;
    box-shadow: 0 4px 20px var(--shadow) !important;
    backdrop-filter: blur(10px);
}
.stDataFrame {
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 4px 20px var(--shadow) !important;
}

/* ═══════════════════════════════════
   HEADINGS
═══════════════════════════════════ */
h2, h3,
h2 *, h3 * {
    font-family: 'Playfair Display', serif !important;
    color: #0d5a7a !important;
    font-weight: 700 !important;
}
/* Streamlit uses p tags inside stMarkdown for subheaders sometimes */
.stMarkdown h2, .stMarkdown h3 {
    color: #0d5a7a !important;
}
.section-label {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #1a7fa8 !important;
    font-weight: 800;
    margin-bottom: 6px;
    opacity: 1 !important;
}
/* General text inside main area - keep dark */
.main p, .main span, .main div {
    color: #0d3a52;
}
/* Dataframe text */
.stDataFrame * {
    color: #0d3a52 !important;
}

/* ═══════════════════════════════════
   HR
═══════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1.5px solid rgba(58,159,214,0.18) !important;
    margin: 2rem 0 !important;
}

/* z-index fix for content */
.main .block-container {
    position: relative;
    z-index: 2;
    padding-top: 0.5rem;
}
</style>

<!-- ══════════════════════════════════
     RICE STALKS (left + right corners)
══════════════════════════════════ -->
<div class="rice-left">
  <div class="rice-stalk">🌾</div>
  <div class="rice-stalk">🌿</div>
  <div class="rice-stalk">🌾</div>
  <div class="rice-stalk">🌿</div>
</div>
<div class="rice-right">
  <div class="rice-stalk">🌾</div>
  <div class="rice-stalk">🌿</div>
  <div class="rice-stalk">🌾</div>
  <div class="rice-stalk">🌿</div>
</div>

<!-- Floating grain particles -->
<div class="grain-float" style="left:7%;  bottom:-60px; animation-duration:9s;  animation-delay:0s;">🌾</div>
<div class="grain-float" style="left:17%; bottom:-60px; animation-duration:12s; animation-delay:2s;">⬭</div>
<div class="grain-float" style="left:82%; bottom:-60px; animation-duration:10s; animation-delay:1s;">🌾</div>
<div class="grain-float" style="left:91%; bottom:-60px; animation-duration:8s;  animation-delay:3.5s;">⬭</div>
<div class="grain-float" style="left:50%; bottom:-60px; animation-duration:14s; animation-delay:5s;">🌾</div>
<div class="grain-float" style="left:35%; bottom:-60px; animation-duration:11s; animation-delay:7s;">⬭</div>
<div class="grain-float" style="left:68%; bottom:-60px; animation-duration:9s;  animation-delay:4s;">🌾</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  LOAD MODEL
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 22px;'>
        <div style='font-size:3rem; animation: stalksway 3s ease-in-out infinite alternate;
                    display:inline-block; filter:drop-shadow(0 3px 10px rgba(255,255,255,0.25));'>🌾</div>
        <div style='font-family:Playfair Display,serif; font-size:1.18rem;
                    color:white; font-weight:700; letter-spacing:0.03em; margin-top:8px;'>
            Rice Grading System
        </div>
        <div style='font-size:0.66rem; color:rgba(255,255,255,0.5);
                    letter-spacing:0.18em; text-transform:uppercase; margin-top:3px;'>
            Control Panel
        </div>
    </div>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.5,
        help="Minimum confidence score for a detection to be counted"
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    mode = st.radio("Inspection Mode", ["📷 Image Analysis", "🎥 Live Inspection"])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:rgba(255,255,255,0.68); line-height:2.0;'>
        <b style='color:rgba(255,255,255,0.95); letter-spacing:0.06em;'>FEATURES</b><br>
        ✔ YOLOv11 detection<br>
        ✔ Multi-class grading<br>
        ✔ Contaminant flagging<br>
        ✔ Purity scoring<br>
        ✔ Batch pass / reject
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  HERO HEADER
# ────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🌾 AI-Powered Quality Inspection</div>
    <h1 class="hero-title">Rice Grading <span>System</span></h1>
    <p class="hero-sub">Automated grain quality classification &amp; contaminant detection</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  LOADING ANIMATION HELPER
# ────────────────────────────────────────────────
LOADING_HTML = """
<div class="loading-scene">
    <div class="loading-person">👨‍🔬</div>
    <div class="conveyor-wrap">
        <div class="grain-belt">🌾</div>
        <div class="grain-belt">⬭</div>
        <div class="grain-belt">🌾</div>
        <div class="conveyor-track"></div>
    </div>
    <div class="loading-bars">
        <div class="loading-bar"></div>
        <div class="loading-bar"></div>
        <div class="loading-bar"></div>
        <div class="loading-bar"></div>
        <div class="loading-bar"></div>
        <div class="loading-bar"></div>
    </div>
    <div class="loading-label">Classifying grains<span class="loading-dots"></span></div>
</div>
"""

# ────────────────────────────────────────────────
#  PROCESS FUNCTION  (RGBA fix included)
# ────────────────────────────────────────────────
def process_image(image_np):
    # Strip alpha channel if RGBA (e.g. PNG screenshots)
    if image_np.ndim == 3 and image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

    results = model(image_np)[0]
    counts  = {"Sound": 0, "Chalky": 0, "Broken": 0, "Unsound": 0, "Contaminants": 0}

    for box in results.boxes:
        cls   = int(box.cls[0])
        label = model.names[cls].lower()
        conf  = float(box.conf[0])
        if conf < conf_threshold:
            continue
        if "sound" in label and "unsound" not in label:
            counts["Sound"] += 1
        elif "chalky" in label:
            counts["Chalky"] += 1
        elif "broken" in label:
            counts["Broken"] += 1
        elif "unsound" in label:
            counts["Unsound"] += 1
        else:
            counts["Contaminants"] += 1

    total     = sum(counts.values())
    purity    = (counts["Sound"] / total * 100) if total > 0 else 0
    annotated = results.plot()
    return annotated, counts, purity, total

# ════════════════════════════════════════════════
#  IMAGE ANALYSIS MODE
# ════════════════════════════════════════════════
if mode == "📷 Image Analysis":

    uploaded_file = st.file_uploader(
        "📂  Drop a rice sample image here",
        type=["jpg", "png", "jpeg"],
        help="JPG / PNG / JPEG — RGBA screenshots are handled automatically"
    )

    if uploaded_file:
        # Force RGB to fix 4-channel RGBA issue
        image    = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Show animated loading scene
        loader = st.empty()
        loader.markdown(LOADING_HTML, unsafe_allow_html=True)
        time.sleep(2.5)

        annotated, counts, purity, total = process_image(image_np)
        loader.empty()

        # ── Image columns ──
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Original Sample</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col2:
            st.markdown('<div class="section-label">AI Detection Output</div>', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)

        st.markdown("---")

        # ── KPI Cards ──
        st.markdown('<div class="section-label">Batch Metrics</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Grains",   total)
        k2.metric("Sound Grains",   counts["Sound"])
        k3.metric("Purity %",       f"{purity:.1f}")
        k4.metric("Contaminants",   counts["Contaminants"])
        is_pass = counts["Contaminants"] == 0 and purity > 80

        # ── Animated status badge ──
        st.markdown("<div style='margin:16px 0 8px;'>", unsafe_allow_html=True)
        if counts["Contaminants"] > 0:
            st.markdown(
                f'<div class="status-fail">❌ {counts["Contaminants"]} Contaminant(s) Detected — Batch Rejected</div>',
                unsafe_allow_html=True
            )
        elif purity < 70:
            st.markdown(
                '<div class="status-warn">⚠️ Low Purity — Manual Inspection Required</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-pass">✅ High Quality Batch — Cleared for Dispatch</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Chart + Table ──
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.subheader("Grain Class Distribution")
            df = pd.DataFrame(list(counts.items()), columns=["Class", "Count"])
            st.bar_chart(df.set_index("Class"), color="#3a9fd6", height=280)
        with col_table:
            st.subheader("Detailed Breakdown")
            df["Share %"] = (df["Count"] / total * 100).round(1) if total > 0 else 0.0
            st.dataframe(
                df.style.format({"Share %": "{:.1f}"}),
                use_container_width=True,
                hide_index=True
            )

# ════════════════════════════════════════════════
#  LIVE CAMERA MODE
# ════════════════════════════════════════════════
elif mode == "🎥 Live Inspection":

    st.subheader("Real-Time Monitoring")
    st.markdown(
        "<p style='color:var(--textSub); font-size:0.92rem;'>"
        "Toggle the camera to begin live grain inspection on your sample tray or conveyor."
        "</p>", unsafe_allow_html=True
    )

    run = st.toggle("▶  Start Camera Feed")
    FRAME_WINDOW = st.image([])

    if "camera" not in st.session_state:
        st.session_state.camera = None

    if run:
        if st.session_state.camera is None:
            cam_loader = st.empty()
            cam_loader.markdown(
                LOADING_HTML.replace("Classifying grains", "Starting camera"),
                unsafe_allow_html=True
            )
            time.sleep(1.4)
            st.session_state.camera = cv2.VideoCapture(0)
            cam_loader.empty()

        cap = st.session_state.camera
        ret, frame = cap.read()

        if ret:
            annotated, counts, purity, total = process_image(frame)
            FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            st.markdown('<div class="section-label">Live Metrics</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Grains",  total)
            c2.metric("Purity %",      f"{purity:.1f}")
            c3.metric("Contaminants",  counts["Contaminants"])
            c4.metric("Status",        "PASS ✅" if counts["Contaminants"] == 0 else "REJECT ❌")

            if counts["Contaminants"] > 0:
                st.markdown(
                    '<div class="status-fail">❌ REJECT SIGNAL ACTIVE</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="status-pass">✅ NORMAL FLOW — All Clear</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ Could not read frame from camera.")

    else:
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
            cv2.destroyAllWindows()
        st.info("📷 Camera is OFF — toggle above to start inspection")

# ── FOOTER ──
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:14px 0 8px;'>
    <div style='font-size:1.8rem;'>🌾</div>
    <div style='font-family:Playfair Display,serif; font-size:1.05rem;
                color:#1a7fa8; font-weight:700; letter-spacing:0.06em; margin-top:4px;'>
        Rice Grading System
    </div>
    <div style='font-size:0.7rem; color:#4a7a96;
                letter-spacing:0.12em; margin-top:5px; text-transform:uppercase;'>
        Developed by Harimohan · ECE · YOLOv11-based Detection
    </div>
</div>
""", unsafe_allow_html=True)
