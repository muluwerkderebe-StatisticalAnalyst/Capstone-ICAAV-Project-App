import streamlit as st
from PIL import Image

from theme import apply_theme

st.set_page_config(
    page_title="ICAAV ML Dashboard",
    page_icon="🚗",
    layout="wide"
)

apply_theme()


# -----------------------------
# BRANDING HEADER
# -----------------------------
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    try:
        st.image("assets/icaav_logo.png", width=120)
    except:
        st.write("ICAAV Logo")

with col2:
    st.markdown("""
        <div class="icaav-page-title">
            Interactive Machine Learning Dashboard for ICAAV Vehicle Data
        </div>

        <div class="icaav-page-subtitle">
            Intelligent Connected Assistive & Autonomous Vehicles (iCAAV) Core
            <br>Advanced Biomechatronics and Locomotion Laboratory
            <br>Carleton University
        </div>
    """, unsafe_allow_html=True)

with col3:
    try:
        st.image("assets/carleton_logo.png", width=120)
    except:
        st.write("Carleton Logo")

st.markdown("---")

# -----------------------------
# INTRO TEXT
# -----------------------------
st.markdown("""
This standalone software platform supports multimodal driving data analysis, 
machine learning model development, visualization, and driver behavior monitoring.

Use the sidebar to navigate between the four major sections:

1. **Data Loading, Feature Engineering, and Visualization**  
2. **Supervised Machine Learning**  
3. **Semi-Supervised and Unsupervised Learning**  
4. **Real-Time Testing, Visualization, and Annotation**  
""")