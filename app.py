import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎯 Field Force Downsizing Simulator", layout="wide", page_icon="")

# =============================================================================
# HELPER: FORMATTAZIONE EUROPEA NUMERI
# =============================================================================
def fmt_eu(value, decimals=0, suffix='', prefix=''):
    if pd.isna(value): return ''
    try:
        num = float(value)
        if decimals == 0: formatted = f"{num:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else: formatted = f"{num:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"{prefix}{formatted}{suffix}"
    except: return str(value)

# =============================================================================
# CONFIGURAZIONE: RETE STRADALE E VELOCITÀ (LIVELLO 1)
# =============================================================================
PROVINCIAL_CIRCUITY = {
    "MI": 1.25, "LO": 1.20, "CR": 1.20, "MN": 1.25, "BS": 1.25, "BG": 1.25, "PV": 1.20,
    "VC": 1.25, "NO": 1.25, "VA": 1.25, "CO": 1.25, "LC": 1.25, "AL": 1.30, "AT": 1.30,
    "BI": 1.25, "VB": 1.40, "TO": 1.30, "CN": 1.30, "RA": 1.25, "FE": 1.25, "PC": 1.25,
    "PR": 1.25, "RE": 1.25, "MO": 1.25, "BO": 1.28, "RN": 1.25, "FC": 1.28, "PU": 1.30,
    "AN": 1.30, "MC": 1.30, "AP": 1.32, "FM": 1.30, "PE": 1.32, "CH": 1.35, "TE": 1.35,
    "AQ": 1.50, "CE": 1.35, "BN": 1.38, "NA": 1.32, "AV": 1.42, "SA": 1.38, "PZ": 1.48,
    "MT": 1.45, "BA": 1.28, "BR": 1.28, "TA": 1.32, "FG": 1.32, "BT": 1.32, "CB": 1.42,
    "IS": 1.42, "VT": 1.38, "LT": 1.32, "FR": 1.32, "RM": 1.32, "RI": 1.38,
    "GE": 1.42, "SV": 1.38, "IM": 1.38, "SP": 1.38, "LU": 1.32, "PI": 1.28, "PT": 1.30,
    "PO": 1.28, "LI": 1.28, "AR": 1.32, "SI": 1.32, "FI": 1.28, "GR": 1.42, "PG": 1.38,
    "TR": 1.32, "MS": 1.35, "UD": 1.32, "GO": 1.30, "TS": 1.28, "PN": 1.32, "VR": 1.25,
    "VI": 1.25, "TV": 1.25, "VE": 1.28, "PD": 1.25, "RO": 1.25, "BL": 1.42, "TN": 1.45,
    "BZ": 1.48, "SO": 1.45, "AO": 1.38,
    "CA": 1.32, "SS": 1.38, "NU": 1.42, "OR": 1.38, "OT": 1.42, "SU": 1.42,
    "RG": 1.42, "SR": 1.38, "CT": 1.32, "ME": 1.38, "PA": 1.38, "TP": 1.42,
    "AG": 1.42, "CL": 1.42, "EN": 1.45, "CS": 1.42, "CZ": 1.42, "VV": 1.48,
    "RC": 1.48, "KR": 1.42, "LE": 1.28
}

PROVINCIAL_SPEED = {
    "MI": 38, "RM": 35, "NA": 32, "TO": 40, "GE": 35, "BO": 38, "FI": 36,
    "VE": 35, "BA": 38, "CT": 34, "PA": 36,
    "LO": 62, "CR": 65, "MN": 60, "PV": 60, "PC": 62, "PR": 62, "RE": 62,
    "MO": 58, "RN": 60, "FE": 60, "RA": 58, "FC": 50, "VR": 60, "VI": 60,
    "PD": 60, "RO": 60, "TV": 60, "BG": 55, "BS": 58, "CO": 50, "VA": 48,
    "NO": 55, "BI": 55, "VC": 55, "AL": 48, "AT": 48, "CN": 50,
    "LU": 55, "PI": 55, "LI": 58, "PO": 55, "PT": 52, "AR": 52, "SI": 52,
    "GR": 48, "LT": 50, "FR": 52, "VT": 50, "RI": 50, "CB": 48, "IS": 48,
    "CE": 45, "BN": 45, "AV": 45, "SA": 48, "PZ": 48, "MT": 48, "FG": 55,
    "BT": 55, "BR": 58, "TA": 55, "LE": 58, "KR": 52,
    "SV": 48, "IM": 45, "SP": 45, "MS": 50, "PG": 50, "TR": 50, "AN": 52,
    "MC": 50, "AP": 48, "FM": 50, "PE": 48, "CH": 45, "TE": 45, "AQ": 42,
    "CS": 45, "CZ": 45, "VV": 42, "RC": 42, "RG": 48, "SR": 48, "EN": 45,
    "CL": 45, "AG": 48, "TP": 48, "ME": 45,
    "AO": 42, "BL": 45, "BZ": 42, "TN": 42, "SO": 42, "UD": 48, "GO": 48,
    "PN": 48, "VB": 45
}

# =============================================================================
# FUNZIONI CORE
# =============================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

def classify_abc(df, volume_col, da_a, da_b, da_c):
    df = df.copy()
    conditions = [
        df[volume_col] >= da_a,
        (df[volume_col] >= da_b) & (df[volume_col] < da_a),
        (df[volume_col] >= da_c) & (df[volume_col] < da_b),
        df[volume_col] < da_c
    ]
    df['classe'] = np.select(conditions, ['A', 'B', 'C', 'Non Attivo'], default='Non Attivo')
    return df

def compute_hull_coords(df_customers):
    if len(df_customers) < 3: return None, None
    try:
        pts = df_customers[['longitudine', 'latitudine']].values
        hull = ConvexHull(pts)
        idx = np.append(hull.vertices, hull.vertices[0])
        return pts[idx, 0], pts[idx, 1]
    except: return None, None

def calculate_travel_metrics(df_customers, rep_home_lat, rep_home_lon, circuity_dict, speed_dict):
    """Livello 1: Calcolo fisico con circuity + velocità provinciale ponderata."""
    if len(df_customers) == 0: return 0.0, 0.0
    
    custs = df_customers.copy()
    air_dists = haversine_km(custs['longitudine'].values, custs['latitudine'].values, rep_home_lon, rep_home_lat)
    custs['air_dist'] = air_dists
    total_visits = custs['freq_visite'].sum()
    if total_visits == 0: return 0.0, 0.0

    custs['circuity'] = custs['sigla'].map(circuity_dict).fillna(1.35)
    custs['road_dist'] = air_dists * custs['circuity']
    avg_road_dist = (custs['road_dist'] * custs['freq_visite']).sum() / total_visits

    custs['speed'] = custs['sigla'].map(speed_dict).fillna(50)
    weighted_speed = (custs['speed'] * custs['freq_visite']).sum() / total_visits

    routing_factor = 1.35  # Simula tour multi-stop giornaliero
    km_annui = total_visits * avg_road_dist * routing_factor
    ore_viaggio = km_annui / weighted_speed if weighted_speed > 0 else 0
    
    return km_annui, ore_viaggio

def run_simulation(df_c, df_v, active_list, col_vol, da_a, da_b, da_c,
                   freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro,
                   max_stops_per_day, use_nearest_neighbor=False):
    
    df_v_act = df_v[df_v['sales rep'].isin(active_list)].copy()
    valid_mask = df_v_act['latitudine'].notna() & df_v_act['longitudine'].notna()
    df_v_valid = df_v_act[valid_mask]
    if len(df_v_valid) == 0: return None, "Errore: nessun venditore selezionato ha coordinate lat/lon valide."
    if len(df_v_valid) < len(active_list):
        missing = set(active_list) - set(df_v_valid['sales rep'])
        st.warning(f"⚠️ {len(missing)} venditori esclusi (coordinate mancanti): {', '.join(list(missing))}")

    df_w = df_c.copy()
    df_w[col_vol] = pd.to_numeric(df_w[col_vol], errors='coerce').fillna(0)
    df_w = classify_abc(df_w, col_vol, da_a, da_b, da_c)
    df_w = df_w[df_w['classe'] != 'Non Attivo'].copy()
    if len(df_w) == 0: return None, "Nessun cliente attivo sopra la soglia minima C."

    if use_nearest_neighbor:
        c_lats, c_lons = df_w['latitudine'].values, df_w['longitudine'].values
        v_lats, v_lons = df_v_valid['latitudine'].values, df_v_valid['longitudine'].values
        valid_rep_names = df_v_valid['sales rep'].values
        dist_matrix = np.zeros((len(c_lats), len(v_lons)))
        for j in range(len(v_lons)):
            dist_matrix[:, j] = haversine_km(c_lons, c_lats, v_lons[j], v_lats[j])
        idx_min = np.argmin(dist_matrix, axis=1)
        min_dists = np.min(dist_matrix, axis=1)
        df_w['assigned_rep'] = valid_rep_names[idx_min]
        df_w['dist_km'] = min_dists
        df_w['assegnazione'] = 'nearest_neighbor'
    else:
        if 'sales rep' not in df_w.columns: return None, "Errore: colonna 'sales rep' mancante nel foglio clienti."
        df_w['assigned_rep'] = df_w['sales rep']
        df_w = df_w[df_w['assigned_rep'].isin(active_list)].copy()
        df_w = df_w.merge(df_v_valid[['sales rep', 'latitudine', 'longitudine']].rename(
            columns={'latitudine': 'rep_lat', 'longitudine': 'rep_lon', 'sales rep': 'assigned_rep'}),
            on='assigned_rep', how='left')
        df_w['dist_km'] = haversine_km(df_w['longitudine'].values, df_w['latitudine'].values,
                                       df_w['rep_lon'].values, df_w['rep_lat'].values)
        df_w['assegnazione'] = 'attuale'
        df_w['rep_lat'] = df_w['assigned_rep'].map(df_v_valid.set_index('sales rep')['latitudine'])
        df_w['rep_lon'] = df_w['assigned_rep'].map(df_v_valid.set_index('sales rep')['longitudine'])

    freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c}
    df_w['freq_visite'] = df_w['classe'].map(freq_map)
    df_w['ore_visita_annue'] = (df_w['freq_visite'] * dur_visita) / 60.0

    travel_data = []
    for rep in active_list:
        sub = df_w[df_w['assigned_rep'] == rep]
        if len(sub) > 0:
            rep_lat = sub['rep_lat'].iloc[0]
            rep_lon = sub['rep_lon'].iloc[0]
            km_totali, ore_viag = calculate_travel_metrics(sub, rep_lat, rep_lon, PROVINCIAL_CIRCUITY, PROVINCIAL_SPEED)
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': ore_viag, 'km_annui': km_totali})
        else:
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': 0.0, 'km_annui': 0.0})

    travel_df = pd.DataFrame(travel_data)
    agg = df_w.groupby('assigned_rep').agg(
        n_clienti=('assigned_rep', 'count'),
        n_clienti_uniq=('sold to id', 'nunique'),
        n_classe_a=('classe', lambda x: (x == 'A').sum()),
        n_classe_b=('classe', lambda x: (x == 'B').sum()),
        n_classe_c=('classe', lambda x: (x == 'C').sum()),
        volume_totale=(col_vol, 'sum'),
        ore_visite_annue=('ore_visita_annue', 'sum')
    ).reset_index().rename(columns={'assigned_rep': 'sales_rep'})

    agg = agg.merge(travel_df, on='sales_rep', how='left').fillna(0)
    max_ore_campo = ore_gg * gg_lavoro
    agg['ore_totali_annue'] = agg['ore_visite_annue'] + agg['ore_viaggio_annue']
    agg['saturazione_pct'] = np.where(max_ore_campo > 0, (agg['ore_totali_annue'] / max_ore_campo) * 100, 0.0)
    agg['driving_min_giorno'] = (agg['ore_viaggio_annue'] * 60) / gg_lavoro
    agg['visite_giorno'] = (agg['ore_visite_annue'] * 60 / dur_visita) / gg_lavoro

    def get_alert(s):
        if s > 110: return "🔴 CRITICO"
        elif s > 100: return "🟠 OVERLOAD"
        elif s > 85: return "⚠️ ATTENZIONE"
        else: return "🟢 OK"
    agg['stato'] = agg['saturazione_pct'].apply(get_alert)

    all_reps = pd.DataFrame({'sales_rep': active_list})
    result = all_reps.merge(agg, on='sales_rep', how='left').fillna(0)
    result = result.sort_values('saturazione_pct', ascending=False)
    return result, df_w

# =============================================================================
# INTERFACCIA
# =============================================================================
def main():
    st.markdown("""
    <style>
    .stDataFrame [data-testid="stDataFrame"] table td, .dataframe td, .dataframe th {
        text-align: center !important; vertical-align: middle !important; justify-content: center !important;
    }
    [data-testid="stMetric"] { text-align: center !important; }
    [data-testid="stMetricValue"] { text-align: center !important; justify-content: center !important; }
    [data-testid="stMetricLabel"] { text-align: center !important; justify-content: center !important; }
    .sidebar-button {
        position: sticky; top: 10px; z-index: 999; background: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;
    }
    .stNumberInput > div > div > input { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Field Force Downsizing Simulator")
    st.markdown("*Simulatore strategico per ottimizzazione rete vendita Italia*")
    
    with st.sidebar:
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        manual_run = st.button("🚀 LANCIA SIMULAZIONE", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("📥 Dati di Input")
        uploaded = st.file_uploader("Carica Excel (Clienti + Venditori)", type=['xlsx'])
        if not uploaded:
            st.info("👆 Carica il file Excel per iniziare")
            return

        file_signature = f"{uploaded.name}_{uploaded.size}"
        if file_signature != st.session_state.get('last_upload_signature', ''):
            st.session_state.last_upload_signature = file_signature
            st.session_state.trigger_auto_run = True
            st.session_state.scenarios = {}
            st.session_state.current_result = None
            st.session_state.current_df_work = None

        try:
            df_c = pd.read_excel(uploaded, sheet_name="clienti_geocodificati")
            df_v = pd.read_excel(uploaded, sheet_name="venditori")
        except Exception as e:
            st.error(f"❌ Errore lettura file: {e}")
            return

        # FIX CRITICO: str(c) evita AttributeError se Excel ha colonne numeriche
        df_c.columns = [str(c).strip().lower() for c in df_c.columns]
        df_v.columns = [str(c).strip().lower() for c in df_v.columns]
        for col in ['latitudine', 'longitudine', 'sigla']:
            if col not in df_c.columns:
                st.error(f"❌ Clienti: manca colonna '{col}'")
                return
        for col in ['latitudine', 'longitudine', 'sales rep']:
            if col not in df_v.columns:
                st.error(f"❌ Venditori: manca colonna '{col}'")
                return
        for col in ['latitudine', 'longitudine']:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            df_v[col] = pd.to_numeric(df_v[col], errors='coerce')
        df_c = df_c.dropna(subset=['latitudine', 'longitudine'])
        if 'sales rep' not in df_c.columns:
            st.error("❌ Colonna 'sales rep' mancante nel foglio clienti.")
            return
        st.success(f"✅ {len(df_c):,} clienti, {len(df_v):,} venditori caricati")
        clienti_con_rep = df_c['sales rep'].notna().sum()
        st.caption(f"📍 {clienti_con_rep:,} clienti hanno un sales rep assegnato")
        st.divider()
        st.subheader("🎮 Modalità")
        modo = st.radio("Scegli modalità:",
            ["📍 Mappa Attuale (Assegnazione Reale)", "🔄 Simula Downsizing (Nearest Neighbor)"], index=0)
        use_nn = (modo == "🔄 Simula Downsizing (Nearest Neighbor)")
        st.divider()
        st.subheader("⚙️ Parametri Simulazione")
        vol_cols = [c for c in df_c.columns if any(k in c.lower() for k in ['gy', 'du', 'tot', '25', '26', 'vol', 'pezzi'])]
        col_vol = st.selectbox("Colonna Volume", vol_cols if vol_cols else df_c.columns.tolist())
        dur_visita = st.slider("⏱️ Durata media visita (min)", 40, 150, 90, step=5)
        ore_gg = st.number_input("🕒 Ore lavorative/giorno", 6.0, 10.0, 8.0, step=0.5)
        gg_lavoro = st.number_input("📅 Giorni lavorativi/anno", 180, 260, 220, step=5)
        pausa_pranzo = st.slider("🍽️ Pausa pranzo (min/giorno)", 0, 120, 60, step=5)
        ore_effettive_gg = ore_gg - (pausa_pranzo / 60.0)
        st.caption(f"*Capacità annua: {ore_effettive_gg * gg_lavoro:,.0f} ore*")
        max_stops_per_day = st.slider("📦 Max visite/giorno", 3, 10, 5, step=1)
        
        st.subheader("👥 Stato Venditori")
        reps = sorted(df_v['sales rep'].unique())
        with st.expander("Attiva / Disattiva venditori", expanded=True):
            rep_status = {r: st.checkbox(r, value=True, key=f"rep_{r}") for r in reps}

        # =============================================================================
        # MATRICE ABC COMPLETA (CON MIN E MAX) - ESATTAMENTE COME ORIGINALE
        # =============================================================================
        st.subheader("📊 Matrice Classificazione ABC & Frequenze")
        st.caption("Definisci gli intervalli esatti (da/a) e le visite annue per ogni classe.")
        if 'abc_vals' not in st.session_state:
            st.session_state.abc_vals = {
                'da_a': 801, 'a_a': -1, 'freq_a': 24,
                'da_b': 301, 'a_b': 800, 'freq_b': 12,
                'da_c': 10, 'a_c': 300, 'freq_c': 3,
                'da_na': 0, 'a_na': 9, 'freq_na': 0
            }
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**🟢 Classe A**")
            da_a = st.number_input("da ≥", key="da_a_input", value=st.session_state.abc_vals['da_a'], min_value=0, step=1)
            a_a = st.number_input("a <", key="a_a_input", value=st.session_state.abc_vals['a_a'], min_value=-1, step=1, help="-1 = infinito")
            freq_a = st.number_input("Visite/anno", key="freq_a_input", value=st.session_state.abc_vals['freq_a'], min_value=0, step=1)
        with col2:
            st.markdown("**🟡 Classe B**")
            da_b = st.number_input("da ≥", key="da_b_input", value=st.session_state.abc_vals['da_b'], min_value=0, step=1)
            a_b = st.number_input("a <", key="a_b_input", value=st.session_state.abc_vals['a_b'], min_value=-1, step=1, help="-1 = infinito")
            freq_b = st.number_input("Visite/anno", key="freq_b_input", value=st.session_state.abc_vals['freq_b'], min_value=0, step=1)
        with col3:
            st.markdown("**🔴 Classe C**")
            da_c = st.number_input("da ≥", key="da_c_input", value=st.session_state.abc_vals['da_c'], min_value=0, step=1)
            a_c = st.number_input("a <", key="a_c_input", value=st.session_state.abc_vals['a_c'], min_value=-1, step=1, help="-1 = infinito")
            freq_c = st.number_input("Visite/anno", key="freq_c_input", value=st.session_state.abc_vals['freq_c'], min_value=0, step=1)
        with col4:
            st.markdown("**🔵 Non Attivi**")
            da_na = st.number_input("da ≥", key="da_na_input", value=st.session_state.abc_vals['da_na'], min_value=0, step=1)
            a_na = st.number_input("a <", key="a_na_input", value=st.session_state.abc_vals['a_na'], min_value=-1, step=1, help="-1 = infinito")
            freq_na = st.number_input("Visite/anno", key="freq_na_input", value=st.session_state.abc_vals['freq_na'], min_value=0, step=1)
        st.session_state.abc_vals = {
            'da_a': da_a, 'a_a': a_a, 'freq_a': freq_a,
            'da_b': da_b, 'a_b': a_b, 'freq_b': freq_b,
            'da_c': da_c, 'a_c': a_c, 'freq_c': freq_c,
            'da_na': da_na, 'a_na': a_na, 'freq_na': freq_na
        }
        min_vol = da_c
        
        if uploaded:
            try:
                df_preview = df_c.copy()
                df_preview[col_vol] = pd.to_numeric(df_preview[col_vol], errors='coerce').fillna(0)
                df_preview = classify_abc(df_preview, col_vol, da_a, da_b, da_c)
                dist = df_preview['classe'].value_counts()
                col_prev1, col_prev2, col_prev3, col_prev4 = st.columns(4)
                with col_prev1: st.metric("🟢 Classe A", f"{fmt_eu(dist.get('A', 0))}")
                with col_prev2: st.metric("🟡 Classe B", f"{fmt_eu(dist.get('B', 0))}")
                with col_prev3: st.metric("🔴 Classe C", f"{fmt_eu(dist.get('C', 0))}")
                with col_prev4: st.metric("🔵 Non Attivi", f"{fmt_eu(dist.get('Non Attivo', 0))}")
            except: pass

    # =============================================================================
    # STATO SESSIONE & LOGICA
    # =============================================================================
    if 'scenarios' not in st.session_state: st.session_state.scenarios = {}
    if 'current_result' not in st.session_state: st.session_state.current_result = None
    if 'current_df_work' not in st.session_state: st.session_state.current_df_work = None
    if 'current_params' not in st.session_state: st.session_state.current_params = {}

    run_sim = False
    if st.session_state.get('trigger_auto_run', False):
        st.session_state.trigger_auto_run = False
        run_sim = True
    if manual_run: run_sim = True

    if run_sim:
        with st.spinner("🔄 Calcolo scenario Density-Aware in corso..."):
            try:
                active_list = [r for r, s in rep_status.items() if s]
                if len(active_list) == 0:
                    st.error("⚠️ Seleziona almeno un venditore")
                else:
                    res, df_w = run_simulation(df_c, df_v, active_list, col_vol, da_a, da_b, da_c,
                                               freq_a, freq_b, freq_c, dur_visita, ore_effettive_gg, gg_lavoro,
                                               max_stops_per_day, use_nearest_neighbor=use_nn)
                    if res is None: st.error(df_w)
                    else:
                        st.session_state.current_result = res
                        st.session_state.current_df_work = df_w
                        st.session_state.current_params = {
                            'active_list': active_list, 'ore_gg': ore_effettive_gg,
                            'gg_lavoro': gg_lavoro, 'dur_visita': dur_visita,
                            'max_stops': max_stops_per_day, 'reps': reps, 'use_nn': use_nn,
                            'modo': modo, 'min_vol': min_vol
                        }
                        st.session_state.current_df_v = df_v
                        st.success("✅ Calcolo completato!")
            except Exception as e:
                st.error(f"❌ Errore: {e}")

    # =============================================================================
    # VISUALIZZAZIONE RISULTATI
    # =============================================================================
    if st.session_state.current_result is not None:
        res = st.session_state.current_result
        df_w = st.session_state.current_df_work
        params = st.session_state.current_params
        df_v_curr = st.session_state.get('current_df_v', df_v)
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Venditori Attivi</div><div style='font-size: 36px; font-weight: bold;'>{fmt_eu(len(params['active_list']))}</div></div>", unsafe_allow_html=True)
        with col2:
            total_customers = int(res['n_clienti'].sum())
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Clienti Serviti</div><div style='font-size: 36px; font-weight: bold;'>{fmt_eu(total_customers)}</div></div>", unsafe_allow_html=True)
        with col3:
            avg_sat = res['saturazione_pct'].mean()
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Saturazione Media</div><div style='font-size: 36px; font-weight: bold;'>{avg_sat:.1f}%</div></div>", unsafe_allow_html=True)
        with col4:
            overload = len(res[res['saturazione_pct'] > 100])
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Venditori Overload</div><div style='font-size: 36px; font-weight: bold;'>{fmt_eu(overload)}</div></div>", unsafe_allow_html=True)
        st.info(f"📍 Modalità: **{params['modo']}** | Stop/Giorno: **{params['max_stops']}** | Soglia minima: **≥{fmt_eu(params.get('min_vol', da_c))}**")
        
        col_btn, col_name = st.columns([2, 1])
        with col_btn:
            if st.button("💾 Salva come Baseline", use_container_width=True):
                st.session_state.scenarios["📍 BASELINE"] = {'result': res.copy(), 'df_work': df_w.copy(), 'params': params}
                st.success("✅ Baseline salvata!")
        with col_name:
            nome_scen = st.text_input("Nome scenario", placeholder="Es. 19 Agenti")
            if st.button("💾 Salva", use_container_width=True) and nome_scen:
                st.session_state.scenarios[nome_scen] = {'result': res.copy(), 'df_work': df_w.copy(), 'params': params}
                st.success(f"✅ '{nome_scen}' salvato!")

        if len(st.session_state.scenarios) >= 2:
            st.divider()
            st.subheader("📊 Confronto Scenari")
            sel = st.selectbox("Confronta con baseline", list(st.session_state.scenarios.keys())[1:])
            base = st.session_state.scenarios["📍 BASELINE"]
            other = st.session_state.scenarios[sel]
            c1, c2, c3 = st.columns(3)
            with c1:
                delta = len(other['result']) - len(base['result'])
                st.metric("Venditori", f"{len(base['result'])} → {len(other['result'])}", f"{delta:+d}")
            with c2:
                st.metric("Clienti", f"{int(base['result']['n_clienti'].sum()):,} → {int(other['result']['n_clienti'].sum()):,}")
            with c3:
                st.metric("Sat. Media", f"{base['result']['saturazione_pct'].mean():.1f}% → {other['result']['saturazione_pct'].mean():.1f}%")

        st.divider()
        st.subheader("📋 Dettaglio Scenario Corrente")
        disp = res[['sales_rep', 'stato', 'n_clienti', 'n_clienti_uniq', 'n_classe_a', 'n_classe_b', 'n_classe_c',
                    'volume_totale', 'ore_visite_annue', 'ore_viaggio_annue', 'ore_totali_annue',
                    'saturazione_pct', 'driving_min_giorno', 'visite_giorno']].copy()
        disp.columns = ['Venditore', 'Stato', 'Clienti', 'Unici', 'A', 'B', 'C', 'Volume',
                        'Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Sat %', 'Min/GG', 'Vis/GG']
        disp_fmt = disp.copy()
        for col in ['Clienti', 'Unici', 'A', 'B', 'C', 'Volume']:
            disp_fmt[col] = disp_fmt[col].apply(lambda x: fmt_eu(x, 0))
        for col in ['Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Min/GG']:
            disp_fmt[col] = disp_fmt[col].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Sat %'] = disp_fmt['Sat %'].apply(lambda x: fmt_eu(x, 1, '%'))
        disp_fmt['Vis/GG'] = disp_fmt['Vis/GG'].apply(lambda x: fmt_eu(x, 2))
        def color_sat(v):
            if pd.isna(v): return ''
            try:
                clean = str(v).replace('%', '').replace('.', '').replace(',', '.')
                n = float(clean)
                if n > 110: return 'background-color:#ffcdd2;color:#b71c1c'
                elif n > 100: return 'background-color:#ffe0b2;color:#e65100'
                elif n > 85: return 'background-color:#fff9c4;color:#f57f17'
                else: return 'background-color:#c8e6c9;color:#1b5e20'
            except: return ''
        styled = disp_fmt.style.map(color_sat, subset=['Sat %']).set_properties(**{'text-align': 'center'})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🗺️ Mappa Territori e Distribuzione Clienti")
        df_map = df_w.dropna(subset=['latitudine', 'longitudine', 'assigned_rep'])
        if len(df_map) == 0:
            st.warning("⚠️ Nessun cliente da visualizzare")
        else:
            st.caption(f"Visualizzati {fmt_eu(len(df_map))} clienti. Le zone colorate rappresentano l'area operativa effettiva di ogni venditore.")
            fig = go.Figure()
            colors = px.colors.qualitative.Alphabet
            rep_colors = {r: colors[i % len(colors)] for i, r in enumerate(params['active_list'])}
            for rep in params['active_list']:
                sub = df_map[df_map['assigned_rep'] == rep]
                if len(sub) >= 3:
                    lon_h, lat_h = compute_hull_coords(sub)
                    if lon_h is not None:
                        fig.add_trace(go.Scattermapbox(
                            mode='lines', lon=lon_h, lat=lat_h,
                            line=dict(width=1.5, color=rep_colors[rep]),
                            fill='toself', fillcolor=rep_colors[rep],
                            opacity=0.25, name=f"Zona {rep}", hoverinfo='name'
                        ))
            classe_colors = {'A': '#ff0000', 'B': '#ffa500', 'C': '#0088ff'}
            for classe, color, size in [('A', classe_colors['A'], 6), ('B', classe_colors['B'], 5), ('C', classe_colors['C'], 4)]:
                df_c = df_map[df_map['classe'] == classe]
                if len(df_c) > 0:
                    fig.add_trace(go.Scattermapbox(
                        lat=df_c['latitudine'], lon=df_c['longitudine'],
                        mode='markers', marker=dict(size=size, color=color, opacity=0.8),
                        name=f"Classe {classe}", hoverdata={'assigned_rep': True}
                    ))
            df_v_active = df_v_curr[df_v_curr['sales rep'].isin(params['active_list'])].dropna(subset=['latitudine', 'longitudine'])
            if len(df_v_active) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=df_v_active['latitudine'], lon=df_v_active['longitudine'],
                    mode='markers', marker=dict(size=14, symbol='star', color='black', line=dict(width=2, color='white')),
                    name=' Home Base', hoverinfo='name'
                ))
            fig.update_layout(
                mapbox_style="carto-positron", mapbox_zoom=5.5, mapbox_center=dict(lat=42.5, lon=12.5),
                margin=dict(r=0, t=30, l=0, b=120), height=650,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                            bgcolor='rgba(255,255,255,0.95)', bordercolor='gray', borderwidth=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("💾 Export Dati")
        export_df = res.copy()
        export_df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        csv = export_df.to_csv(index=False, sep=';', decimal=',')
        st.download_button("📥 Scarica Report CSV", csv, f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
    else:
        st.info("📥 Carica Excel e clicca '🚀 Lancia Simulazione' per iniziare.")

if __name__ == "__main__":
    main()
