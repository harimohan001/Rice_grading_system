import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="RiceGuard AI Enterprise",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------------------------------
# PREMIUM UI STYLING
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#07111d,
#0e2438,
#08131f
);
color:white;
}

.block-container{
padding-top:1rem;
}

h1,h2,h3,h4,p,label,span{
color:white !important;
}

section[data-testid="stSidebar"]{
background:#111d2d;
}

[data-testid="metric-container"]{
background:rgba(255,255,255,.06);
padding:22px;
border-radius:20px;
border:1px solid rgba(255,255,255,.08);
box-shadow:0px 8px 24px rgba(0,0,0,.25);
}

[data-testid="stFileUploader"]{
background:#162638;
padding:20px;
border-radius:18px;
}

.hero{
padding:50px;
border-radius:30px;
background:linear-gradient(
135deg,
rgba(0,198,255,.18),
rgba(0,114,255,.07)
);
border:1px solid rgba(255,255,255,.08);
}

.footer{
text-align:center;
padding:30px;
font-size:18px;
color:#cfd9e6;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model=load_model()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙ RiceGuard Control")

conf_threshold=st.sidebar.slider(
"Detection Confidence",
0.1,1.0,0.5
)

st.sidebar.markdown("---")

st.sidebar.success("""
Plant Status : ONLINE

Vision Engine : ACTIVE

Sorting Logic : READY
""")


# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown("""
<div class='hero'>
<h1 style='font-size:58px;'>
🌾 RiceGuard AI Enterprise
</h1>

<h3>
Industrial Vision Intelligence for
Rice Quality Grading
</h3>

Real-Time Inspection • Batch Analytics • Contaminant Detection
</div>
""",unsafe_allow_html=True)

st.write("")


# ---------------------------------------------------
# DETECTION FUNCTION
# ---------------------------------------------------

def process_image(img):

    results=model(img)[0]

    counts={
        "Sound":0,
        "Chalky":0,
        "Broken":0,
        "Unsound":0,
        "Contaminants":0
    }

    for box in results.boxes:

        cls=int(box.cls[0])
        label=model.names[cls].lower()

        conf=float(box.conf[0])

        if conf<conf_threshold:
            continue

        if label=="sound":
            counts["Sound"]+=1

        elif "chalky" in label:
            counts["Chalky"]+=1

        elif "broken" in label:
            counts["Broken"]+=1

        elif "unsound" in label:
            counts["Unsound"]+=1

        else:
            counts["Contaminants"]+=1

    total=sum(counts.values())

    purity=(
        counts["Sound"]/total*100
        if total>0 else 0
    )

    if purity>=85:
        grade="Premium"

    elif purity>=70:
        grade="Commercial"

    else:
        grade="Reject"

    annotated=results.plot()

    return annotated,counts,purity,total,grade


# ---------------------------------------------------
# UPLOAD
# ---------------------------------------------------

uploaded=st.file_uploader(
"Upload Rice Sample Image",
type=["jpg","jpeg","png"]
)

if uploaded:

    img=Image.open(uploaded)
    img_np=np.array(img)

    with st.spinner("Running Inspection..."):
        time.sleep(1)
        annotated,counts,purity,total,grade=process_image(img_np)

    st.success("Inspection Completed")

    # --------------------------------
    # IMAGES
    # --------------------------------

    c1,c2=st.columns(2)

    with c1:
        st.subheader("Input Sample")
        st.image(
            img,
            use_container_width=True
        )

    with c2:
        st.subheader("AI Detection Output")
        st.image(
            annotated,
            use_container_width=True
        )

    st.markdown("---")

    # --------------------------------
    # KPI DASHBOARD
    # --------------------------------

    st.subheader("Plant Analytics Dashboard")

    k1,k2,k3,k4=st.columns(4)

    k1.metric(
        "Detected Grains",
        total
    )

    k2.metric(
        "Purity %",
        f"{purity:.2f}"
    )

    k3.metric(
        "Quality Grade",
        grade
    )

    k4.metric(
        "Contaminants",
        counts["Contaminants"]
    )


    st.markdown("---")

    # --------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------

    st.subheader("Class Distribution")

    df=pd.DataFrame(
        list(counts.items()),
        columns=["Class","Count"]
    )

    st.bar_chart(
        df.set_index("Class")
    )


    st.markdown("---")

    # --------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------

    st.subheader("Executive Quality Summary")

    a,b,c=st.columns(3)

    a.metric(
        "Defect %",
        f"{100-purity:.2f}"
    )

    b.metric(
        "Chalky Count",
        counts["Chalky"]
    )

    c.metric(
        "Broken Count",
        counts["Broken"]
    )


    if counts["Contaminants"]>0:
        st.error(
"⚠ Foreign contaminants detected. Secondary sorting recommended."
        )

    elif purity>=85:
        st.success(
"✓ Premium export-grade batch."
        )

    else:
        st.warning(
"Inspection review recommended."
        )


    # =============================================
    # DIGITAL TWIN CONVEYOR
    # =============================================

    accepted=counts["Sound"]
    rejected=counts["Contaminants"]+counts["Unsound"]

    st.markdown("---")
    st.subheader("🏭 Digital Twin Sorting Conveyor")

    st.markdown(f"""
<style>

.conveyor {{
position:relative;
height:120px;
background:#263b50;
border-radius:60px;
overflow:hidden;
margin-top:30px;
}}

.belt {{
position:absolute;
width:200%;
height:100%;
background:
repeating-linear-gradient(
90deg,
#304a63 0px,
#304a63 50px,
#223547 50px,
#223547 100px
);
animation:beltmove 4s linear infinite;
}}

@keyframes beltmove {{
0%{{transform:translateX(0);}}
100%{{transform:translateX(-50%);}}
}}

.grain {{
position:absolute;
top:48px;
width:16px;
height:16px;
background:#ffd966;
border-radius:50%;
animation:flow 8s linear infinite;
}}

.g1{{left:-30px;animation-delay:0s;}}
.g2{{left:-130px;animation-delay:1s;}}
.g3{{left:-230px;animation-delay:2s;}}
.g4{{left:-330px;animation-delay:3s;}}

@keyframes flow {{
0%{{left:-30px;}}
100%{{left:105%;}}
}}

.binwrap {{
display:flex;
justify-content:space-around;
margin-top:35px;
}}

.bin {{
width:280px;
padding:25px;
border-radius:18px;
font-size:24px;
font-weight:bold;
text-align:center;
}}

.good {{
background:#0f4930;
}}

.bad {{
background:#4a1e1e;
}}

</style>

<div class='conveyor'>
<div class='belt'></div>

<div class='grain g1'></div>
<div class='grain g2'></div>
<div class='grain g3'></div>
<div class='grain g4'></div>

<div style='position:absolute;left:50%;top:22px;font-size:38px;'>
🤖
</div>

<div style='position:absolute;left:75%;top:18px;font-size:38px;'>
🚨
</div>

</div>

<div class='binwrap'>

<div class='bin good'>
Accepted Bin<br><br>
{accepted}
</div>

<div class='bin bad'>
Reject Bin<br><br>
{rejected}
</div>

</div>
""",unsafe_allow_html=True)



# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div class='footer'>
RiceGuard AI Enterprise Suite • Industrial Deployment Dashboard
</div>
""",unsafe_allow_html=True)
