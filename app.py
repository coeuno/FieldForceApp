import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Strategic Downsizing Simulator", layout="wide")

st.markdown("""
    <style>
    .kpi-container { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9; }
    .small-font { font-size: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Strategic Downsizing Simulator")

# --- FUNZIONI ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

# --- CARICAMENTO ---
uploaded_file = st.sidebar.file_uploader("Carica File Excel", type=["xlsx"])

if uploaded_file:
    df_c = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
    df_v = pd.read_excel(uploaded_file, sheet_name="venditori")
    
    # PULIZIA TOTALE: minuscolo e strip spazi
    df_c.columns = [c.strip().lower() for c in df_c.columns]
    df_v.columns = [c.strip().lower() for c in df_v.columns]
    
    # Mappatura nomi colonne: assicura che esistano
    df_c = df_c.rename(columns={'sales rep': 'sales rep', 'sigla': 'sigla', 'latitudine': 'latitudine', 'longitudine': 'longitudine'})
    df_v = df_v.rename(columns={'sales rep': 'sales rep', 'latitudine': 'latitudine', 'longitudine': 'longitudine'})

    # --- SIDEBAR ---
    st.sidebar.subheader("👤 Venditori Attivi")
    v_list = sorted(df_v['sales rep'].unique())
    # Lista compatta
    active_reps = {v: st.sidebar.checkbox(v, True, key=v) for v in v_list}
    
    if st.sidebar.button("🚀 AVVIA SIMULAZIONE"):
        st.session_state.running = True

    # --- LOGICA CALCOLO ---
    # Se non è in simulazione, usa i dati originali (Baseline)
    if not st.session_state.get('running', False):
        df_display = df_c.copy()
        titolo = "Situazione Attuale (Baseline)"
    else:
        active_list = [v for v, status in active_reps.items() if status]
        df_active_v = df_v[df_v['sales rep'].isin(active_list)]
        
        c_coords = df_c[['longitudine', 'latitudine']].values
        v_coords = df_active_v[['longitudine', 'latitudine']].values
        
        dists = haversine(c_coords[:,0][:,None], c_coords[:,1][:,None], v_coords[:,0][None,:], v_coords[:,1][None,:])
        idx = np.argmin(dists, axis=1)
        
        df_display = df_c.copy()
        df_display['assigned_rep'] = [active_list[i] for i in idx]
        titolo = "Scenario Simulato"

    # --- VISUALIZZAZIONE ---
    st.subheader(titolo)
    
    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Venditori Attivi", len(df_display['sales rep'].unique() if 'assigned_rep' not in df_display else df_display['assigned_rep'].unique()))
    col2.metric("Totale Clienti", len(df_display))
    
    # Tabella sintesi
    st.write("Sintesi per Venditore:")
    sintesi = df_display.groupby('assigned_rep' if 'assigned_rep' in df_display else 'sales rep').size().reset_index(name='clienti')
    st.dataframe(sintesi, use_container_width=True)

    # Mappa
    fig = px.scatter_mapbox(df_display, lat="latitudine", lon="longitudine", color='assigned_rep' if 'assigned_rep' in df_display else 'sales rep', 
                            zoom=5, height=500, render_mode="webgl")
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Carica il file Excel per visualizzare la situazione attuale.")
