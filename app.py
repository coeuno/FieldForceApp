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
    if pd.isna(value):
        return ''
    try:
        num = float(value)
        if decimals == 0:
            formatted = f"{num:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            formatted = f"{num:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"{prefix}{formatted}{suffix}"
    except:
        return str(value)


# =============================================================================
# CONFIGURAZIONE
# =============================================================================
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
    "CA": 1.25, "SU": 1.25, "OT": 1.25,
    "AO": 1.40, "BL": 1.40, "BZ": 1.40, "TN": 1.40, "SO": 1.40, "AL": 1.40, "AT": 1.40, "AQ": 1.40
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

def compute_hull(df, lat_c, lon_c):
    if len(df) < 3:
        return None, None
    try:
        pts = df[[lon_c, lat_c]].values
        hull = ConvexHull(pts)
        idx = np.append(hull.vertices, hull.vertices[0])
        return pts[idx, 0], pts[idx, 1]
    except:
        return None, None

def run_simulation(df_c, df_v, active_list, col_vol, da_a, da_b, da_c,
                   freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro, vel_media,
                   use_nearest_neighbor=False):
    df_v_act = df_v[df_v['sales rep'].isin(active_list)].copy()
    valid_mask = df_v_act['latitudine'].notna() & df_v_act['longitudine'].notna()
    df_v_valid = df_v_act[valid_mask]

    if len(df_v_valid) == 0:
        return None, "Errore: nessun venditore selezionato ha coordinate lat/lon valide."

    if len(df_v_valid) < len(active_list):
        missing = set(active_list) - set(df_v_valid['sales rep'])
        st.warning(f"⚠️ {len(missing)} venditori esclusi (coordinate mancanti): {', '.join(list(missing))}")

    df_w = df_c.copy()
    df_w[col_vol] = pd.to_numeric(df_w[col_vol], errors='coerce').fillna(0)
    df_w = classify_abc(df_w, col_vol, da_a, da_b, da_c)
    df_w = df_w[df_w['classe'] != 'Non Attivo'].copy()

    if len(df_w) == 0:
        return None, "Nessun cliente attivo sopra la soglia minima C."

    df_w['tortuosity'] = df_w['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)

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
        if 'sales rep' not in df_w.columns:
            return None, "Errore: colonna 'sales rep' mancante nel foglio clienti."

        df_w['assigned_rep'] = df_w['sales rep']
        df_w = df_w[df_w['assigned_rep'].isin(active_list)].copy()

        df_w = df_w.merge(
            df_v_valid[['sales rep', 'latitudine', 'longitudine']].rename(
                columns={'latitudine': 'rep_lat', 'longitudine': 'rep_lon', 'sales rep': 'assigned_rep'}
            ),
            on='assigned_rep',
            how='left'
        )
        df_w['dist_km'] = haversine_km(
            df_w['longitudine'].values, df_w['latitudine'].values,
            df_w['rep_lon'].values, df_w['rep_lat'].values
        )
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
            dists = haversine_km(sub['longitudine'].values, sub['latitudine'].values, rep_lon, rep_lat)
            max_radius = np.max(dists) if len(dists) > 0 else 0
            avg_dist = max_radius * 0.55
            avg_tort = sub['tortuosity'].mean()
            total_visits = sub['freq_visite'].sum()
            ore_viag = (total_visits * avg_dist * avg_tort) / vel_media
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': ore_viag})
        else:
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': 0.0})

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

    agg['saturazione_pct'] = np.where(
        max_ore_campo > 0,
        (agg['ore_totali_annue'] / max_ore_campo) * 100,
        0.0
    )
    agg['driving_min_giorno'] = (agg['ore_viaggio_annue'] * 60) / gg_lavoro
    agg['visite_giorno'] = (agg['ore_visite_annue'] * 60 / dur_visita) / gg_lavoro

    def get_alert(s):
        if s > 110:
            return "🔴 CRITICO"
        elif s > 100:
            return "🟠 OVERLOAD"
        elif s > 85:
            return "🟡 ATTENZIONE"
        else:
            return "🟢 OK"

    agg['stato'] = agg['saturazione_pct'].apply(get_alert)

    all_reps = pd.DataFrame({'sales_rep': active_list})
    result = all_reps.merge(agg, on='sales_rep', how='left').fillna(0)
    result = result.sort_values('saturazione_pct', ascending=False)

    return result, df_w


# =============================================================================
# INTERFACCIA
# =============================================================================
def main():
    # =============================================================================
    # CSS PERSONALIZZATO - FORZA ALLINEAMENTO CENTRALE
    # =============================================================================
    st.markdown("""
    <style>
        /* FORZA allineamento centro per TUTTE le tabelle Streamlit */
        .stDataFrame [data-testid="stDataFrame"] table td,
        .stDataFrame [data-testid="stDataFrame"] table th,
        div[data-testid="stDataFrame"] table td,
        div[data-testid="stDataFrame"] table th,
        .dataframe td, 
        .dataframe th {
            text-align: center !important;
            vertical-align: middle !important;
            justify-content: center !important;
        }
        
        /* Allinea metric al centro */
        [data-testid="stMetric"] {
            text-align: center !important;
        }
        [data-testid="stMetricValue"] {
            text-align: center !important;
            justify-content: center !important;
        }
        [data-testid="stMetricLabel"] {
            text-align: center !important;
            justify-content: center !important;
        }
        
        /* Sidebar pulsante fisso in alto */
        .sidebar-button {
            position: sticky;
            top: 10px;
            z-index: 999;
            background: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        /* Input fields centrati */
        .stNumberInput > div > div > input {
            text-align: center !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Field Force Downsizing Simulator")
    st.markdown("*Simulatore strategico per ottimizzazione rete vendita Italia*")

    # --- SIDEBAR ---
    with st.sidebar:
        # PULSANTE FISSO IN ALTO
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        manual_run = st.button("🚀 LANCIA SIMULAZIONE", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("📁 Dati di Input")
        uploaded = st.file_uploader("Carica Excel (Clienti + Venditori)", type=['xlsx'])

        if not uploaded:
            st.info("👆 Carica il file per iniziare")
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

        df_c.columns = [c.strip().lower() for c in df_c.columns]
        df_v.columns = [c.strip().lower() for c in df_v.columns]

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

        # --- MODALITÀ ---
        st.subheader("🎮 Modalità")
        modo = st.radio(
            "Scegli modalità:",
            ["📍 Mappa Attuale (Assegnazione Reale)", "🔄 Simula Downsizing (Nearest Neighbor)"],
            index=0
        )
        use_nn = (modo == "🔄 Simula Downsizing (Nearest Neighbor)")
        st.divider()

        # --- PARAMETRI ---
        st.subheader("⚙️ Parametri Simulazione")

        vol_cols = [c for c in df_c.columns if any(k in c.lower() for k in ['gy', 'du', 'tot', '25', '26', 'vol', 'pezzi'])]
        col_vol = st.selectbox("Colonna Volume", vol_cols if vol_cols else df_c.columns.tolist())

        dur_visita = st.slider("⏱️ Durata media visita (min)", 40, 150, 90, step=5)
        ore_gg = st.number_input("🕒 Ore lavorative/giorno", 6.0, 10.0, 8.0, step=0.5, 
                                 help="Ore totali disponibili (incluse guida e pause)")
        gg_lavoro = st.number_input("📅 Giorni lavorativi/anno (netti)", 180, 260, 220, step=5)
        
        pausa_pranzo = st.slider("🍽️ Pausa pranzo (min/giorno)", 0, 120, 60, step=15,
                                 help="Tempo sottratto dalle ore lavorative per la pausa pranzo")
        
        ore_effettive_gg = ore_gg - (pausa_pranzo / 60.0)
        st.caption(f"*Capacità annua effettiva: {ore_effettive_gg * gg_lavoro:,.0f} ore (dopo pausa)*")

        vel_media = st.slider("🚗 Velocità media (km/h)", 40, 100, 65, step=5)

        # --- VENDITORI ---
        st.subheader("👥 Stato Venditori")
        reps = sorted(df_v['sales rep'].unique())
        with st.expander("Attiva / Disattiva venditori", expanded=True):
            rep_status = {r: st.checkbox(r, value=True, key=f"rep_{r}") for r in reps}

    # =============================================================================
    # MATRICE ABC - INPUT DIRETTI (NO DATA_EDITOR)
    # =============================================================================
    st.subheader("📊 Matrice Classificazione ABC & Frequenze")
    st.caption("Modifica soglie e visite direttamente nei campi sotto.")

    # Inizializzazione valori default
    if 'abc_values' not in st.session_state:
        st.session_state.abc_values = {
            'da_a': 801, 'a_a': -1, 'freq_a': 24,
            'da_b': 301, 'a_b': 800, 'freq_b': 12,
            'da_c': 10, 'a_c': 300, 'freq_c': 3,
            'da_na': 0, 'a_na': 9, 'freq_na': 0
        }

    # Layout a griglia con 4 colonne
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🟢 Classe A**")
        da_a = st.number_input("da ≥", key="da_a", value=st.session_state.abc_values['da_a'], min_value=0, step=1)
        a_a = st.number_input("a <", key="a_a", value=st.session_state.abc_values['a_a'], min_value=-1, step=1)
        freq_a = st.number_input("Visite/anno", key="freq_a", value=st.session_state.abc_values['freq_a'], min_value=0, step=1)
    
    with col2:
        st.markdown("**🟡 Classe B**")
        da_b = st.number_input("da ≥", key="da_b", value=st.session_state.abc_values['da_b'], min_value=0, step=1)
        a_b = st.number_input("a <", key="a_b", value=st.session_state.abc_values['a_b'], min_value=-1, step=1)
        freq_b = st.number_input("Visite/anno", key="freq_b", value=st.session_state.abc_values['freq_b'], min_value=0, step=1)
    
    with col3:
        st.markdown("**🔴 Classe C**")
        da_c = st.number_input("da ≥", key="da_c", value=st.session_state.abc_values['da_c'], min_value=0, step=1)
        a_c = st.number_input("a <", key="a_c", value=st.session_state.abc_values['a_c'], min_value=-1, step=1)
        freq_c = st.number_input("Visite/anno", key="freq_c", value=st.session_state.abc_values['freq_c'], min_value=0, step=1)
    
    with col4:
        st.markdown("**🔵 Non Attivi**")
        da_na = st.number_input("da ≥", key="da_na", value=st.session_state.abc_values['da_na'], min_value=0, step=1)
        a_na = st.number_input("a <", key="a_na", value=st.session_state.abc_values['a_na'], min_value=-1, step=1)
        freq_na = st.number_input("Visite/anno", key="freq_na", value=st.session_state.abc_values['freq_na'], min_value=0, step=1)

    # Salva valori aggiornati
    st.session_state.abc_values = {
        'da_a': da_a, 'a_a': a_a, 'freq_a': freq_a,
        'da_b': da_b, 'a_b': a_b, 'freq_b': freq_b,
        'da_c': da_c, 'a_c': a_c, 'freq_c': freq_c,
        'da_na': da_na, 'a_na': a_na, 'freq_na': freq_na
    }

    min_vol = da_c

    if not (da_a > da_b > da_c >= da_na):
        st.warning("⚠️ Le soglie dovrebbero essere: A > B > C ≥ Non Attivi")

    # Preview distribuzione clienti
    if uploaded:
        try:
            df_preview = df_c.copy()
            df_preview[col_vol] = pd.to_numeric(df_preview[col_vol], errors='coerce').fillna(0)
            df_preview = classify_abc(df_preview, col_vol, da_a, da_b, da_c)
            dist = df_preview['classe'].value_counts()
            total = len(df_preview)
            total_attivi = len(df_preview[df_preview['classe'] != 'Non Attivo'])

            # Layout centrato con columns
            col_prev1, col_prev2, col_prev3, col_prev4 = st.columns(4)
            
            with col_prev1:
                n_a = dist.get('A', 0)
                st.markdown(f"<div style='text-align: center;'><span style='color: #00ff00; font-size: 24px;'>●</span><br><b>Classe A (≥{fmt_eu(da_a, 0)})</b></div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold;'>{fmt_eu(n_a, 0)}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; color: #00ff00;'>↑ {n_a/total*100:.1f}%</div>", 
                           unsafe_allow_html=True)
                
            with col_prev2:
                n_b = dist.get('B', 0)
                st.markdown(f"<div style='text-align: center;'><span style='color: #ffff00; font-size: 24px;'>●</span><br><b>Classe B ({fmt_eu(da_b, 0)}-{fmt_eu(da_a-1, 0)})</b></div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold;'>{fmt_eu(n_b, 0)}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; color: #00ff00;'>↑ {n_b/total*100:.1f}%</div>", 
                           unsafe_allow_html=True)
                
            with col_prev3:
                n_c = dist.get('C', 0)
                st.markdown(f"<div style='text-align: center;'><span style='color: #ff0000; font-size: 24px;'>●</span><br><b>Classe C ({fmt_eu(da_c, 0)}-{fmt_eu(da_b-1, 0)})</b></div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold;'>{fmt_eu(n_c, 0)}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; color: #00ff00;'>↑ {n_c/total*100:.1f}%</div>", 
                           unsafe_allow_html=True)
                
            with col_prev4:
                n_na = dist.get('Non Attivo', 0)
                st.markdown(f"<div style='text-align: center;'><span style='color: #0088ff; font-size: 24px;'>●</span><br><b>Non Attivi (<{fmt_eu(da_c, 0)})</b></div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 36px; font-weight: bold;'>{fmt_eu(n_na, 0)}</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; color: #00ff00;'>↑ {n_na/total*100:.1f}%</div>", 
                           unsafe_allow_html=True)

            st.markdown(f"<div style='text-align: center; margin-top: 10px; padding: 10px; background-color: #1e1e1e; border-radius: 5px;'>📊 Clienti attivi (A+B+C): <b>{fmt_eu(total_attivi, 0)}</b> su {fmt_eu(total, 0)} totali ({total_attivi/total*100:.1f}%)</div>", 
                       unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Preview non disponibile: {e}")

    # =============================================================================
    # STATO SESSIONE
    # =============================================================================
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_df_work' not in st.session_state:
        st.session_state.current_df_work = None
    if 'current_params' not in st.session_state:
        st.session_state.current_params = {}

    # =============================================================================
    # LOGICA DI LANCIO
    # =============================================================================
    run_sim = False

    if st.session_state.get('trigger_auto_run', False):
        st.session_state.trigger_auto_run = False
        run_sim = True

    if manual_run:
        run_sim = True

    if run_sim:
        with st.spinner("⚡ Calcolo scenario in corso..."):
            try:
                active_list = [r for r, s in rep_status.items() if s]
                if len(active_list) == 0:
                    st.error("⚠️ Seleziona almeno un venditore")
                else:
                    res, df_w = run_simulation(
                        df_c, df_v, active_list, col_vol, da_a, da_b, da_c,
                        freq_a, freq_b, freq_c, dur_visita, ore_effettive_gg, gg_lavoro, vel_media,
                        use_nearest_neighbor=use_nn
                    )
                    if res is None:
                        st.error(df_w)
                    else:
                        st.session_state.current_result = res
                        st.session_state.current_df_work = df_w
                        st.session_state.current_params = {
                            'active_list': active_list,
                            'ore_gg': ore_effettive_gg,
                            'gg_lavoro': gg_lavoro,
                            'vel_media': vel_media,
                            'dur_visita': dur_visita,
                            'reps': reps,
                            'use_nn': use_nn,
                            'modo': modo,
                            'min_vol': min_vol
                        }
                        st.session_state.current_df_v = df_v
                        st.success(f"✅ Simulazione completata! Clienti sotto {fmt_eu(min_vol, 0)} esclusi (Non Attivi).")
            except Exception as e:
                st.error(f"❌ Errore calcolo: {e}")
                import traceback
                st.code(traceback.format_exc())

    # =============================================================================
    # VISUALIZZAZIONE RISULTATI
    # =============================================================================
    if st.session_state.current_result is not None:
        res = st.session_state.current_result
        df_w = st.session_state.current_df_work
        params = st.session_state.current_params
        df_v_curr = st.session_state.get('current_df_v', df_v)

        # --- KPI ---
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Venditori Attivi</div><div style='font-size: 36px; font-weight: bold;'>{fmt_eu(len(params['active_list']), 0)}</div></div>", 
                       unsafe_allow_html=True)
        with col2:
            total_customers = int(res['n_clienti'].sum())
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Clienti Serviti (Righe)</div><div style='font-size: 36px; font-weight: bold;'>{fmt_eu(total_customers, 0)}</div></div>", 
                       unsafe_allow_html=True)
        with col3:
            avg_sat = res['saturazione_pct'].mean()
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Saturazione Media</div><div style='font-size: 36px; font-weight: bold;'>{avg_sat:.1f}%</div></div>", 
                       unsafe_allow_html=True)
        with col4:
            overload = len(res[res['saturazione_pct'] > 100])
            st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Venditori Overload</div><div style='font-size: 36px; font-weight: bold;'>{fmt_eu(overload, 0)}</div></div>", 
                       unsafe_allow_html=True)

        st.info(f"📍 Modalità: **{params['modo']}** | Soglia minima: **≥{fmt_eu(params.get('min_vol', da_c), 0)}** | " + 
                ("Assegnazione dal file clienti" if not params['use_nn'] else "Assegnazione ricalcolata con Nearest Neighbor"))

        # --- SALVATAGGIO SCENARI ---
        col_btn, col_name = st.columns([2, 1])
        with col_btn:
            if st.button("💾 Salva come Baseline (Attuale)", use_container_width=True):
                st.session_state.scenarios["📍 BASELINE (Attuale)"] = {
                    'result': res.copy(),
                    'df_work': df_w.copy(),
                    'params': params
                }
                st.success("✅ Baseline salvata!")

        with col_name:
            nome_scen = st.text_input("Nome nuovo scenario", placeholder="Es. 19 Agenti")
            if st.button("💾 Salva Scenario", use_container_width=True) and nome_scen:
                st.session_state.scenarios[nome_scen] = {
                    'result': res.copy(),
                    'df_work': df_w.copy(),
                    'params': params
                }
                st.success(f"✅ '{nome_scen}' salvato!")

        # --- CONFRONTO SCENARI ---
        if len(st.session_state.scenarios) >= 2:
            st.divider()
            st.subheader("📊 Confronto Scenari")
            base_name = "📍 BASELINE (Attuale)"
            altri = [k for k in st.session_state.scenarios.keys() if k != base_name]
            if altri:
                sel_scen = st.selectbox("Confronta con baseline", altri)
                if sel_scen in st.session_state.scenarios:
                    base = st.session_state.scenarios[base_name]
                    other = st.session_state.scenarios[sel_scen]

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        delta_v = len(other['result']) - len(base['result'])
                        st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Venditori</div><div style='font-size: 24px; font-weight: bold;'>{fmt_eu(len(base['result']), 0)} → {fmt_eu(len(other['result']), 0)}</div><div style='color: {'red' if delta_v < 0 else 'green'}; font-size: 18px;'>{'↓' if delta_v < 0 else '↑'} {fmt_eu(abs(delta_v), 0)}</div></div>", 
                                   unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Clienti</div><div style='font-size: 24px; font-weight: bold;'>{fmt_eu(int(base['result']['n_clienti'].sum()), 0)} → {fmt_eu(int(other['result']['n_clienti'].sum()), 0)}</div></div>", 
                                   unsafe_allow_html=True)
                    with c3:
                        base_sat = base['result']['saturazione_pct'].mean()
                        other_sat = other['result']['saturazione_pct'].mean()
                        delta_sat = other_sat - base_sat
                        st.markdown(f"<div style='text-align: center;'><div style='font-size: 14px; color: #888;'>Sat. Media</div><div style='font-size: 24px; font-weight: bold;'>{base_sat:.1f}% → {other_sat:.1f}%</div><div style='color: {'red' if delta_sat > 5 else 'green' if delta_sat < -5 else 'orange'}; font-size: 18px;'>{'↑' if delta_sat > 0 else '↓'} {abs(delta_sat):.1f}%</div></div>", 
                                   unsafe_allow_html=True)

                    df_base = base['result'].set_index('sales_rep')
                    df_other = other['result'].set_index('sales_rep')
                    delta = df_other[['saturazione_pct', 'n_clienti', 'ore_totali_annue']].subtract(
                        df_base[['saturazione_pct', 'n_clienti', 'ore_totali_annue']], fill_value=0
                    ).reset_index()
                    delta.columns = ['Venditore', 'Δ Saturazione %', 'Δ Clienti', 'Δ Ore Totali']
                    
                    delta_fmt = delta.copy()
                    delta_fmt['Δ Saturazione %'] = delta_fmt['Δ Saturazione %'].apply(lambda x: fmt_eu(x, 1, '%'))
                    delta_fmt['Δ Clienti'] = delta_fmt['Δ Clienti'].apply(lambda x: fmt_eu(x, 0))
                    delta_fmt['Δ Ore Totali'] = delta_fmt['Δ Ore Totali'].apply(lambda x: fmt_eu(x, 1))
                    
                    st.dataframe(
                        delta_fmt.style.set_properties(**{'text-align': 'center'})
                        .background_gradient(subset=['Δ Saturazione %'], cmap='RdYlGn_r', vmin=-50, vmax=50),
                        use_container_width=True,
                        hide_index=True
                    )

        # --- TABELLA DETTAGLIO ---
        st.divider()
        st.subheader("📋 Dettaglio Scenario Corrente")

        disp = res[['sales_rep', 'stato', 'n_clienti', 'n_clienti_uniq', 'n_classe_a', 'n_classe_b', 'n_classe_c',
                    'volume_totale', 'ore_visite_annue', 'ore_viaggio_annue', 'ore_totali_annue',
                    'saturazione_pct', 'driving_min_giorno', 'visite_giorno']].copy()
        disp.columns = ['Venditore', 'Stato', 'Clienti (Righe)', 'Clienti Unici', 'A', 'B', 'C', 'Volume',
                        'Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Sat %', 'Min/GG', 'Vis/GG']

        disp_fmt = disp.copy()
        disp_fmt['Clienti (Righe)'] = disp_fmt['Clienti (Righe)'].apply(lambda x: fmt_eu(x, 0))
        disp_fmt['Clienti Unici'] = disp_fmt['Clienti Unici'].apply(lambda x: fmt_eu(x, 0))
        disp_fmt['A'] = disp_fmt['A'].apply(lambda x: fmt_eu(x, 0))
        disp_fmt['B'] = disp_fmt['B'].apply(lambda x: fmt_eu(x, 0))
        disp_fmt['C'] = disp_fmt['C'].apply(lambda x: fmt_eu(x, 0))
        disp_fmt['Volume'] = disp_fmt['Volume'].apply(lambda x: fmt_eu(x, 0))
        disp_fmt['Ore Visite'] = disp_fmt['Ore Visite'].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Ore Viaggio'] = disp_fmt['Ore Viaggio'].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Ore Totali'] = disp_fmt['Ore Totali'].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Sat %'] = disp_fmt['Sat %'].apply(lambda x: fmt_eu(x, 1, '%'))
        disp_fmt['Min/GG'] = disp_fmt['Min/GG'].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Vis/GG'] = disp_fmt['Vis/GG'].apply(lambda x: fmt_eu(x, 2))

        def color_sat(v):
            if pd.isna(v):
                return ''
            try:
                clean = str(v).replace('%', '').replace('.', '').replace(',', '.')
                n = float(clean)
                if n > 110:
                    return 'background-color:#ffcdd2;color:#b71c1c'
                elif n > 100:
                    return 'background-color:#ffe0b2;color:#e65100'
                elif n > 85:
                    return 'background-color:#fff9c4;color:#f57f17'
                else:
                    return 'background-color:#c8e6c9;color:#1b5e20'
            except:
                return ''

        styled = disp_fmt.style.map(color_sat, subset=['Sat %']).set_properties(**{'text-align': 'center'})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- MAPPA ---
        st.subheader("🗺️ Mappa Territori")
        df_map = df_w.dropna(subset=['latitudine', 'longitudine', 'assigned_rep'])

        if len(df_map) == 0:
            st.warning("⚠️ Nessun cliente valido da visualizzare sulla mappa.")
        else:
            st.caption(f"Visualizzati {fmt_eu(len(df_map), 0)} clienti su {fmt_eu(len(df_w), 0)} totali")
            show_hull = st.checkbox("Mostra confini territori", value=True)

            fig = px.scatter_mapbox(
                df_map,
                lat="latitudine",
                lon="longitudine",
                color="assigned_rep",
                size="freq_visite",
                size_max=10,
                zoom=5,
                height=600,
                opacity=0.8,
                hover_name="assigned_rep",
                hover_data={
                    "classe": True,
                    "freq_visite": True,
                    "dist_km": ":.1f"
                }
            )

            if show_hull:
                colors = px.colors.qualitative.Set3
                for i, r in enumerate(params['active_list']):
                    sub = df_map[df_map['assigned_rep'] == r]
                    if len(sub) >= 3:
                        lon_h, lat_h = compute_hull(sub, 'latitudine', 'longitudine')
                        if lon_h is not None:
                            fig.add_trace(go.Scattermapbox(
                                mode="lines",
                                lon=lon_h,
                                lat=lat_h,
                                line=dict(width=2, color=colors[i % len(colors)]),
                                name=f"Confine {r}",
                                showlegend=True
                            ))

            df_v_active = df_v_curr[df_v_curr['sales rep'].isin(params['active_list'])].dropna(
                subset=['latitudine', 'longitudine']
            )
            if len(df_v_active) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=df_v_active['latitudine'],
                    lon=df_v_active['longitudine'],
                    mode='markers+text',
                    marker=dict(size=16, symbol='star', color='black', line=dict(width=2, color='white')),
                    text=df_v_active['sales rep'],
                    textposition="top center",
                    textfont=dict(size=10, color='black'),
                    name='🏠 Home Base',
                    hoverinfo='text'
                ))

            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- EXPORT ---
        st.divider()
        st.subheader("💾 Export Dati")
        export_df = res.copy()
        export_df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        export_df['modalita'] = params['modo']
        csv = export_df.to_csv(index=False, sep=';', decimal=',')
        st.download_button(
            "📥 Scarica Report CSV",
            csv,
            f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )

    else:
        st.info("👈 Carica il file Excel nella sidebar e clicca '🚀 Lancia Simulazione' per iniziare.")


if __name__ == "__main__":
    main()
