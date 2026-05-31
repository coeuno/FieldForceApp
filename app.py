import streamlit as st
import pandas as pd

# Configurazione della pagina (a tutto schermo)
st.set_page_config(page_title="Ottimizzazione Territori", layout="wide")

st.title("🎯 Field Force Optimization - Motore di Simulazione")

# --- 1. BARRA LATERALE (I tuoi comandi) ---
st.sidebar.header("⚙️ Parametri di Simulazione")

st.sidebar.subheader("1. Carica i Dati")
file_venditori = st.sidebar.file_uploader("Carica Excel Venditori", type=['xlsx'])
file_clienti = st.sidebar.file_uploader("Carica Excel Clienti", type=['xlsx'])

st.sidebar.subheader("2. Regole di Lavoro")
giorni_lavorativi = st.sidebar.slider("Giorni lavorativi annui", min_value=150, max_value=250, value=220, step=5)
minuti_visita = st.sidebar.slider("Minuti medi per visita", min_value=15, max_value=120, value=45, step=5)

st.sidebar.subheader("3. Matrice Visite")
visite_A = st.sidebar.number_input("Visite annue Classe A (> 500 pz)", value=12)
visite_B = st.sidebar.number_input("Visite annue Classe B (100-500 pz)", value=6)
visite_C = st.sidebar.number_input("Visite annue Classe C (< 100 pz)", value=2)

# --- 2. SCHERMATA CENTRALE ---
if file_venditori is not None and file_clienti is not None:
    st.success("✅ File caricati con successo! Il sistema è pronto.")
    
    # Leggiamo i file Excel appena caricati
    df_ven = pd.read_excel(file_venditori)
    df_cli = pd.read_excel(file_clienti)
    
    # Mostriamo un'anteprima per rassicurare l'utente
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"👤 Trovati **{len(df_ven)} venditori**.")
        st.dataframe(df_ven.head(3)) # Mostra le prime 3 righe
        
    with col2:
        st.info(f"🏪 Trovati **{len(df_cli)} clienti**.")
        st.dataframe(df_cli.head(3)) # Mostra le prime 3 righe
        
    st.warning("🚧 I prossimi moduli (Calcolo Distanze e Raggruppamento Mappa) verranno costruiti qui nei prossimi step!")
else:
    st.info("👈 Inizia caricando i due file Excel dalla barra laterale a sinistra.")
