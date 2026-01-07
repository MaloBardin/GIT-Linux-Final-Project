# streamlit_app.py
import streamlit as st
st.set_page_config(layout="wide")
page_cible_url = "/home_page"

st.markdown(
    f'<meta http-equiv="refresh" content="0; url={page_cible_url}" target="_self">',
    unsafe_allow_html=True
)

st.stop()


def barre_menu():
    # On met 2 colonnes pour aligner les boutons côte à côte
    col1, col2 = st.columns(2)

    with col1:
        # Lien vers la page principale (le fichier actuel)
        st.page_link("accueil.py", label="🏠 Accueil", use_container_width=True)

    with col2:
        # Lien vers la page dans le dossier 'pages'
        # ATTENTION : Le fichier doit exister dans le dossier 'pages' !
        st.page_link("pages/ma_page_2.py", label="🚀 Vers Page 2", use_container_width=True)

    st.divider()  # Une ligne pour séparer


# --- L'AFFICHAGE DE LA PAGE ---
st.set_page_config(page_title="Accueil", layout="wide")

# 1. On affiche le menu
barre_menu()

# 2. Le contenu de la page
st.title("Bienvenue sur l'Accueil")
st.write("Clique sur le bouton à droite pour changer vraiment de page.")