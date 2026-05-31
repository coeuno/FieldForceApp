import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import yaml

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Strategic Downsizing Simulator", layout="wide")

# CSS Custom per pulizia visiva
st.markdown("""
    <style>
    .main-title { font-size: 24px !important; font-weight: bold; margin-bottom: 20px; }
    .kpi-box { border: 1px solid #ddd; padding: 15px; border-radius: 10px; background-color: #f9f9f9; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">Strategic Downsizing Simulator & Force Redistribution</p>', unsafe_allow_html=True)

# --- DIZIONARIO PROVINCE (Da usare con colonna SIGLA) ---
PROVINCIAL_TORTUOSITY = {
    "MI": 1.15, "LO": 1.15, "CR": 1.15, "MN": 1.15, "BS": 1.15, "BG": 1.15, "PV": 1.15,
    "VC": 1.15, "NO": 1.15, "VE": 1.15, "PD": 1.15, "RO": 1.15, "VR": 1.15, "VI": 1.15,
    "TV": 1.15, "FE": 1.15, "RN": 1.15, "LI": 1.15, "PI": 1.15, "PO": 1.15, "LU": 1.15,
    "MS": 1.15, "PR": 1.15, "PC": 1.15, "RE": 1.15, "MO": 1.15, "BO": 1.15, "RA": 1.15,
    "FC": 1.15, "PU": 1.15, "FG": 1.15, "BT": 1.15, "BA": 1.15, "BR": 1.15, "LE": 1.15,
    "TA": 1.15, "RM": 1.15, "LT": 1.15, "FR": 1.15,
    "VA": 1.25, "CO": 1.25, "LC": 1.25, "SP": 1.25, "IM": 1.25, "SV": 1.25, "CN": 1.25,
    "GE": 1.25, "AN": 1.25, "MC": 1.25, "FM": 1.25, "AP": 1.25, "PE": 1.25, "CH": 1.25,
    "TE": 1.25, "RI": 1.25, "VT": 1.25, "GR": 1.25, "PT": 1.25, "FI": 1.25, "SI": 1.25,
    "AR": 1.25, "PG": 1.25, "TR": 1.25, "CB": 1.25, "IS": 1.25, "CE": 1.25, "NA": 1.25,
    "AV": 1.25, "BN": 1.25, "SA": 1.25, "PZ": 1.25, "MT": 1.25, "CS": 1.25, "CZ": 1.25,
    "VV": 1.25, "RC": 1.25, "TP": 1.25, "PA": 1.25, "ME": 1.25, "CT": 1.25, "SR": 1.25,
    "RG": 1.25, "AG": 1.25, "CL": 1.25, "EN": 1.25, "SS": 1.25, "NU": 1.25, "OR": 1.25,
    "CA": 1.25, "SU": 1.25, "AO": 1.40, "BL": 1.40, "BZ": 1.40, "TN": 1.40, "SO": 1.40, 
    "AL": 1.40, "AT": 1.40, "AQ": 1.40
}

# --- FUNZIONI DI CALCOLO ---
def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

# --- SIDEBAR E INPUT ---
uploaded_file = st.sidebar.file_uploader("Carica Excel", type=["xlsx"])
if uploaded_file:
    df_c = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
    df_v = pd.read_excel(uploaded_file, sheet_name="venditori")
    
    # Normalizzazione nomi (già puliti da te, ma doppia sicurezza)
    df_c.columns = [c.strip().lower() for c in df_c.columns]
    df_v.columns = [c.strip().lower() for c in df_v.columns]
    
    # Mappatura Province
    df_c['tortuosity'] = df_c['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)
    
    # Check integrità venditori
    v_list = df_v['sales rep'].unique()
    missing = set(df_c['sales rep'].unique()) - set(v_list)
    if missing: st.warning(f"⚠️ Venditori non trovati nel foglio 'venditori': {missing}")

    # Configurazione parametri
    st.sidebar.subheader("🎛️ Parametri Simulazione")
    metric = st.sidebar.selectbox("Metrica Volume", [c for c in df_c.columns if '26' in c or '25' in c])
    h_visit = st.sidebar.number_input("Ore per visita", 1.5)
    speed = st.sidebar.slider("Velocità media (km/h)", 40, 110, 70)
    
    # Stato Venditori (Checklist)
    st.sidebar.subheader("👤 Venditori Attivi")
    active_reps = {v: st.sidebar.checkbox(v, True) for v in v_list}
    
    # Pulsante SIMULA
    if st.sidebar.button("🚀 AVVIA SIMULAZIONE"):
        st.session_state.running = True
    
    if st.session_state.get('running', False):
        # LOGICA RIASSEGNAZIONE
        active_list = [v for v, status in active_reps.items() if status]
        df_active_v = df_v[df_v['sales rep'].isin(active_list)]
        
        # Broadcasting Haversine + Tortuosità
        c_coords = df_c[['longitudine', 'latitudine']].values
        v_coords = df_active_v[['longitudine', 'latitudine']].values
        
        # Calcolo distanze
        dists = haversine_vectorized(
            c_coords[:,0][:,None], c_coords[:,1][:,None],
            v_coords[:,0][None,:], v_coords[:,1][None,:]
        )
        
        # Assegnazione
        idx = np.argmin(dists, axis=1)
        df_c['assigned_rep'] = [active_list[i] for i in idx]
        df_c['dist_km'] = np.min(dists, axis=1) * df_c['tortuosity']
        
        # Validazione
        st.markdown('<div class="kpi-box">', unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("Risorse Attive", f"{len(active_list)} / {len(v_list)}")
        cols[1].metric("Clienti Riassegnati", f"{(df_c['sales rep'] != df_c['assigned_rep']).sum()}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mappa
        fig = px.scatter_mapbox(df_c, lat="latitudine", lon="longitudine", color="assigned_rep", zoom=5, height=500, render_mode="webgl")
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    
else:
    st.info("Carica il file Excel per iniziare.")
