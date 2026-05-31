import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Strategic Downsizing Simulator", layout="wide")

# --- DIZIONARIO TORTUOSITÀ ---
PROVINCIAL_TORTUOSITY = {
    "MI": 1.15, "LO": 1.15, "CR": 1.15, "MN": 1.15, "BS": 1.15, "BG": 1.15, "PV": 1.15,
    "VC": 1.15, "NO": 1.15, "VE": 1.15, "PD": 1.15, "RO": 1.15, "VR": 1.15, "VI": 1.15,
    "TV": 1.15, "FE": 1.15, "RN": 1.15, "LI": 1.15, "PI": 1.15, "PO": 1.15, "LU": 1.15,
    "MS": 1.15, "PR": 1.15, "PC": 1.15, "RE": 1.15, "MO": 1.15, "BO": 1.15, "RA": 1.15,
    "FC": 1.15, "PU": 1.15, "FG": 1.15, "BT": 1.15, "BA": 1.15, "BR": 1.15, "LE": 1.15,
    "TA": 1.15, "RM": 1.15, "LT": 1.15, "FR": 1.15, "VA": 1.25, "CO": 1.25, "LC": 1.25, 
    "SP": 1.25, "IM": 1.25, "SV": 1.25, "CN": 1.25, "GE": 1.25, "AN": 1.25, "MC": 1.25, 
    "FM": 1.25, "AP": 1.25, "PE": 1.25, "CH": 1.25, "TE": 1.25, "RI": 1.25, "VT": 1.25, 
    "GR": 1.25, "PT": 1.25, "FI": 1.25, "SI": 1.25, "AR": 1.25, "PG": 1.25, "TR": 1.25, 
    "CB": 1.25, "IS": 1.25, "CE": 1.25, "NA": 1.25, "AV": 1.25, "BN": 1.25, "SA": 1.25, 
    "PZ": 1.25, "MT": 1.25, "CS": 1.25, "CZ": 1.25, "VV": 1.25, "RC": 1.25, "TP": 1.25, 
    "PA": 1.25, "ME": 1.25, "CT": 1.25, "SR": 1.25, "RG": 1.25, "AG": 1.25, "CL": 1.25, 
    "EN": 1.25, "SS": 1.25, "NU": 1.25, "OR": 1.25, "CA": 1.25, "SU": 1.25, "AO": 1.40, 
    "BL": 1.40, "BZ": 1.40, "TN": 1.40, "SO": 1.40, "AL": 1.40, "AT": 1.40, "AQ": 1.40
}

# --- FUNZIONI ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

# --- CARICAMENTO E PULIZIA ---
uploaded_file = st.sidebar.file_uploader("Carica File Excel", type=["xlsx"])

if uploaded_file:
    df_c = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
    df_v = pd.read_excel(uploaded_file, sheet_name="venditori")
    
    # Normalizzazione nomi colonne e tipi dati (essenziale per evitare KeyError/TypeError)
    df_c.columns = [c.strip().lower() for c in df_c.columns]
    df_v.columns = [c.strip().lower() for c in df_v.columns]
    
    # Forzatura tipi numerici (evita errori Plotly)
    for col in ['latitudine', 'longitudine']:
        df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
        df_v[col] = pd.to_numeric(df_v[col], errors='coerce')
    df_c = df_c.dropna(subset=['latitudine', 'longitudine'])
    
    # --- SIDEBAR PARAMETRI ---
    st.sidebar.subheader("🎛️ Parametri Simulazione")
    col_volume = st.sidebar.selectbox("Colonna Volumi Target", [c for c in df_c.columns if '2' in c or 'tot' in c])
    
    min_vol = st.sidebar.slider("Taglia minima cliente (esclusione)", 0, 100, 0)
    durata_visita = st.sidebar.number_input("Durata visita (ore)", 1.5, step=0.1)
    vel_media = st.sidebar.slider("Velocità media (km/h)", 40, 110, 70)
    
    # Frequenze
    st.sidebar.subheader("📅 Frequenza Visite/Anno")
    freq_a = st.sidebar.number_input("Classe A", 24)
    freq_b = st.sidebar.number_input("Classe B", 12)
    freq_c = st.sidebar.number_input("Classe C", 6)
    
    # Venditori
    st.sidebar.subheader("👤 Venditori Attivi")
    v_list = sorted(df_v['sales rep'].unique())
    active_reps = {v: st.sidebar.checkbox(v, True) for v in v_list}

    # --- CALCOLO SIMULAZIONE ---
    if st.sidebar.button("🚀 LANCIA SIMULAZIONE"):
        active_list = [v for v, status in active_reps.items() if status]
        df_active_v = df_v[df_v['sales rep'].isin(active_list)]
        
        # Filtro volumi
        df_display = df_c[df_c[col_volume] >= min_vol].copy()
        
        # Calcolo distanze e tortuosità
        df_display['tortuosity'] = df_display['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)
        
        c_coords = df_display[['longitudine', 'latitudine']].values
        v_coords = df_active_v[['longitudine', 'latitudine']].values
        
        dists = haversine(c_coords[:,0][:,None], c_coords[:,1][:,None], v_coords[:,0][None,:], v_coords[:,1][None,:])
        idx = np.argmin(dists, axis=1)
        
        df_display['assigned_rep'] = [active_list[i] for i in idx]
        df_display['dist_km'] = np.min(dists, axis=1) * df_display['tortuosity']
        
        # Logica Ore
        df_display['ore_viaggio'] = (df_display['dist_km'] * 2) / vel_media
        df_display['ore_visita'] = durata_visita # (qui si può raffinare con la classe ABC)
        
        # --- OUTPUT ---
        st.subheader("📊 Sintesi KPI Scenario")
        
        # Sintesi per venditore
        sintesi = df_display.groupby('assigned_rep').agg({
            'sales rep': 'count', # Clienti totali
            'ore_viaggio': 'sum',
            'ore_visita': 'sum'
        }).rename(columns={'sales rep': 'n_clienti'})
        sintesi['ore_totali'] = sintesi['ore_viaggio'] + sintesi['ore_visita']
        sintesi['saturazione'] = (sintesi['ore_totali'] / 1760) * 100 # Basato su 1760 ore annue
        
        st.dataframe(sintesi.style.format("{:.1f}").background_gradient(subset=['saturazione'], cmap="YlOrRd"), use_container_width=True)
        
        # Mappa
        fig = px.scatter_mapbox(df_display, lat="latitudine", lon="longitudine", color='assigned_rep', 
                                zoom=5, height=500, render_mode="webgl")
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Configura i parametri e premi 'LANCIA SIMULAZIONE'")

else:
    st.info("Carica il file Excel per iniziare.")
