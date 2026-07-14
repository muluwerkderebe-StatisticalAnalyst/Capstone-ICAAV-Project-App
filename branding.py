"""Shared branding header component for Streamlit pages."""

import streamlit as st
from PIL import Image
from pathlib import Path


def render_header(title: str, subtitle_html: str = "", logo_width: int = 100):
    """
    Render a consistent branding header across all Streamlit pages.
    
    Parameters:
    -----------
    title : str
        Main page title
    subtitle_html : str
        HTML-formatted subtitle text (optional)
    logo_width : int
        Width of logos in pixels (default: 100)
    """
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        try:
            logo_path = Path(__file__).parent / "assets" / "icaav_logo.png"
            st.image(str(logo_path), width=logo_width)
        except Exception:
            st.markdown("**iCAAV**")
    
    with col2:
        st.markdown(f"""
        <div class="icaav-page-title">
            {title}
        </div>

        <div class="icaav-page-subtitle">
            {subtitle_html}
            <br>iCAAV Core · Advanced Biomechatronics and Locomotion Laboratory · Carleton University
        </div>
    """, unsafe_allow_html=True)
    
    with col3:
        try:
            logo_path = Path(__file__).parent / "assets" / "carleton_logo.png"
            st.image(str(logo_path), width=logo_width)
        except Exception:
            st.markdown("**Carleton**")
    
    st.markdown("---")
