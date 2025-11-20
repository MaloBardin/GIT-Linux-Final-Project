import streamlit as st
from pathlib import Path

def local_css():
    css_file_path = Path(__file__).parent.parent / "style.css"
    try:
        with open(css_file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Erreur: Fichier CSS non trouvé à {css_file_path}")

local_css()

