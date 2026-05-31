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
    """
    Classifica clienti in base alle soglie 'da' della tabella.
    A: volume >= da_a
    B: volume >= da_b e volume < da_a
    C: volume >= da_c e volume < da_b
    """
    df = df.copy()
    conditions = [
        df[volume_col] >= da_a,
        (df[volume_col] >= da_b) & (df[volume_col] < da_a),
        df[volume_col] < da_b
    ]
    df['classe'] = np.select(conditions, ['A', 'B', 'C'], default='C')
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

def run_simulation(df_c, df_v, active_list, col_vol, min_vol, da_a, da_b, da_c,
                   freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro, vel_media,
                   use_nearest_neighbor=False):
    """
    Esegue l'intero calcolo di assegnazione e saturazione.
    """
    df_v_act = df_v[df_v['sales rep'].isin(active_list)].copy()
    valid_mask = df_v_act['latitudine'].notna() & df_v_act['longitudine'].notna()
    df_v_valid = df_v_act[valid_mask]

    if len(df_v_valid) == 0:
        return None, "Errore: nessun venditore selezionato ha coordinate lat/lon valide."

    if len(df_v_valid) < len(active_list):
        missing = set(active_list) - set(df_v_valid['sales rep'])
        st.warning(f"⚠️ {len(missing)} venditori esclusi (coordinate mancanti): {', '.join(list(missing))}")

    # --- Preparazione clienti ---
    df_w = df_c.copy()
    df_w[col_vol] = pd.to_numeric(df_w[col_vol], errors='coerce').fillna(0)

    if min_vol > 0:
        df_w = df_w[df_w[col_vol] >= min_vol]

    if len(df_w) == 0:
        return None, "Nessun cliente sopra la soglia minima."

    df_w = classify_abc(df_w, col_vol, da_a, da_b, da_c)
    df_w['tortuosity'] = df_w['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)

    # --- ASSEGNAZIONE CLIENTI ---
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

    # --- Frequenze e ore visita ---
    freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c}
    df_w['freq_visite'] = df_w['classe'].map(freq_map)
    df_w['ore_visita_annue'] = (df_w['freq_visite'] * dur_visita) / 60.0

    # --- Modello viaggio ---
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

    # --- Aggregazione per venditore ---
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
    st.title("🎯 Field Force Downsizing Simulator")
    st.markdown("*Simulatore strategico per ottimizzazione rete vendita Italia*")

    # --- SIDEBAR ---
    with st.sidebar:
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

        min_vol = st.number_input("🔻 Taglia minima cliente", -1, 100000, -1, step=100)

        dur_visita = st.slider("⏱️ Durata media visita (min)", 40, 150, 90, step=5)
        ore_gg = st.number_input("🕒 Ore lavorative/giorno", 6.0, 10.0, 8.0, step=0.5)
        gg_lavoro = st.number_input("📅 Giorni lavorativi/anno (netti)", 180, 260, 220, step=5)
        st.caption(f"*Capacità annua lorda: {ore_gg * gg_lavoro:,.0f} ore*")

        vel_media = st.slider("🚗 Velocità media (km/h)", 40, 100, 65, step=5)

        # --- VENDITORI ---
        st.subheader("👥 Stato Venditori")
        reps = sorted(df_v['sales rep'].unique())
        with st.expander("Attiva / Disattiva venditori", expanded=True):
            rep_status = {r: st.checkbox(r, value=True, key=f"rep_{r}") for r in reps}

        manual_run = st.button("🚀 LANCIA SIMULAZIONE", type="primary", use_container_width=True)

    # =============================================================================
    # MATRICE ABC - STRUTTURA 4 COLONNE (da/a/Categoria/Visite)
    # =============================================================================
    st.subheader("📊 Matrice Classificazione ABC & Frequenze")
    st.caption("Modifica soglie (da/a) e visite per profilare i clienti sulla colonna volume selezionata")

    # RESET FORZATO: se la matrice in session state ha colonne vecchie, la sovrascrivo
    expected_cols = ['da', 'a', 'Categoria', 'Visite anno']

    if 'abc_matrix' not in st.session_state:
        st.session_state.abc_matrix = pd.DataFrame({
            'da': [801, 301, 0],
            'a': ['Max', 800, 300],
            'Categoria': ['A', 'B', 'C'],
            'Visite anno': [24, 12, 1]
        })
    else:
        # Verifica che la struttura sia corretta (4 colonne), altrimenti reset
        existing_cols = list(st.session_state.abc_matrix.columns)
        if existing_cols != expected_cols:
            st.session_state.abc_matrix = pd.DataFrame({
                'da': [801, 301, 0],
                'a': ['Max', 800, 300],
                'Categoria': ['A', 'B', 'C'],
                'Visite anno': [24, 12, 1]
            })

    # data_editor con TUTTE le colonne modificabili (tranne Categoria)
    edited_matrix = st.data_editor(
        st.session_state.abc_matrix,
        column_config={
            'da': st.column_config.NumberColumn(
                label='da',
                step=1, 
                min_value=0,
                help="Soglia minima (inclusiva) per questa categoria"
            ),
            'a': st.column_config.TextColumn(
                label='a',
                help="Soglia massima. Scrivi 'Max' per infinito."
            ),
            'Categoria': st.column_config.TextColumn(
                label='Categoria',
                disabled=True
            ),
            'Visite anno': st.column_config.NumberColumn(
                label='Visite anno',
                step=1, 
                min_value=0,
                help="Numero di visite annuali per questa categoria"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="matrice_abc_v2",  # KEY CAMBIATA per forzare ricreazione
        num_rows="fixed"
    )

    # Salva la matrice modificata in session state
    st.session_state.abc_matrix = edited_matrix.copy()

    # Estrazione parametri dalla matrice con gestione errori robusta
    try:
        # Ordina per Categoria per sicurezza
        edited_matrix = edited_matrix.sort_values('Categoria').reset_index(drop=True)

        da_a = int(float(edited_matrix.loc[edited_matrix['Categoria'] == 'A', 'da'].values[0]))
        da_b = int(float(edited_matrix.loc[edited_matrix['Categoria'] == 'B', 'da'].values[0]))
        da_c = int(float(edited_matrix.loc[edited_matrix['Categoria'] == 'C', 'da'].values[0]))
        freq_a = int(float(edited_matrix.loc[edited_matrix['Categoria'] == 'A', 'Visite anno'].values[0]))
        freq_b = int(float(edited_matrix.loc[edited_matrix['Categoria'] == 'B', 'Visite anno'].values[0]))
        freq_c = int(float(edited_matrix.loc[edited_matrix['Categoria'] == 'C', 'Visite anno'].values[0]))

        # Validazione: le soglie devono essere coerenti (A > B > C)
        if not (da_a > da_b > da_c):
            st.warning("⚠️ Le soglie dovrebbero essere decrescenti: A > B > C")
    except Exception as e:
        st.error(f"❌ Errore nella lettura della matrice: {e}. Uso valori default.")
        da_a, da_b, da_c, freq_a, freq_b, freq_c = 801, 301, 0, 24, 12, 1

    # Preview distribuzione clienti
    if uploaded:
        try:
            df_preview = df_c.copy()
            df_preview[col_vol] = pd.to_numeric(df_preview[col_vol], errors='coerce').fillna(0)
            df_preview = classify_abc(df_preview, col_vol, da_a, da_b, da_c)
            dist = df_preview['classe'].value_counts().sort_index()
            total = len(df_preview)

            col_prev1, col_prev2, col_prev3 = st.columns(3)
            with col_prev1:
                n_a = dist.get('A', 0)
                st.metric(f"🔴 Classe A (≥{da_a:,})", f"{n_a:,}", f"{n_a/total*100:.1f}%")
            with col_prev2:
                n_b = dist.get('B', 0)
                st.metric(f"🟡 Classe B ({da_b:,}-{da_a-1:,})", f"{n_b:,}", f"{n_b/total*100:.1f}%")
            with col_prev3:
                n_c = dist.get('C', 0)
                st.metric(f"🟢 Classe C (<{da_b:,})", f"{n_c:,}", f"{n_c/total*100:.1f}%")
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
                        df_c, df_v, active_list, col_vol, min_vol, da_a, da_b, da_c,
                        freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro, vel_media,
                        use_nearest_neighbor=use_nn
                    )
                    if res is None:
                        st.error(df_w)
                    else:
                        st.session_state.current_result = res
                        st.session_state.current_df_work = df_w
                        st.session_state.current_params = {
                            'active_list': active_list,
                            'ore_gg': ore_gg,
                            'gg_lavoro': gg_lavoro,
                            'vel_media': vel_media,
                            'dur_visita': dur_visita,
                            'reps': reps,
                            'use_nn': use_nn,
                            'modo': modo
                        }
                        st.session_state.current_df_v = df_v
                        st.success("✅ Simulazione completata!")
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
            st.metric("Venditori Attivi", len(params['active_list']))
        with col2:
            total_customers = int(res['n_clienti'].sum())
            st.metric("Clienti Serviti (Righe)", f"{total_customers:,}")
        with col3:
            avg_sat = res['saturazione_pct'].mean()
            st.metric("Saturazione Media", f"{avg_sat:.1f}%")
        with col4:
            overload = len(res[res['saturazione_pct'] > 100])
            st.metric("Venditori Overload", overload, delta_color="inverse")

        st.info(f"📍 Modalità: **{params['modo']}** | " + 
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
                        st.metric("Venditori", f"{len(base['result'])} → {len(other['result'])}", delta=delta_v)
                    with c2:
                        st.metric("Clienti",
                                  f"{int(base['result']['n_clienti'].sum()):,} → {int(other['result']['n_clienti'].sum()):,}")
                    with c3:
                        st.metric("Sat. Media",
                                  f"{base['result']['saturazione_pct'].mean():.1f}% → {other['result']['saturazione_pct'].mean():.1f}%")

                    df_base = base['result'].set_index('sales_rep')
                    df_other = other['result'].set_index('sales_rep')
                    delta = df_other[['saturazione_pct', 'n_clienti', 'ore_totali_annue']].subtract(
                        df_base[['saturazione_pct', 'n_clienti', 'ore_totali_annue']], fill_value=0
                    ).reset_index()
                    delta.columns = ['Venditore', 'Δ Saturazione %', 'Δ Clienti', 'Δ Ore Totali']
                    st.dataframe(
                        delta.style.format({
                            'Δ Saturazione %': '{:+.1f}%',
                            'Δ Clienti': '{:+.0f}',
                            'Δ Ore Totali': '{:+.1f}'
                        }).background_gradient(subset=['Δ Saturazione %'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )

        # --- TABELLA DETTAGLIO ---
        st.divider()
        st.subheader("📋 Dettaglio Scenario Corrente")

        disp = res[['sales_rep', 'stato', 'n_clienti', 'n_clienti_uniq', 'n_classe_a', 'n_classe_b', 'n_classe_c',
                    'volume_totale', 'ore_visite_annue', 'ore_viaggio_annue', 'ore_totali_annue',
                    'saturazione_pct', 'driving_min_giorno', 'visite_giorno']].copy()
        disp.columns = ['Venditore', 'Stato', 'Clienti (Righe)', 'Clienti Unici', 'A', 'B', 'C', 'Volume',
                        'Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Sat %', 'Min/GG', 'Vis/GG']

        def color_sat(v):
            if pd.isna(v):
                return ''
            try:
                n = float(str(v).replace('%', ''))
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

        styled = disp.style.map(color_sat, subset=['Sat %']).format({
            'Volume': '{:,.0f}',
            'Ore Visite': '{:.1f}',
            'Ore Viaggio': '{:.1f}',
            'Ore Totali': '{:.1f}',
            'Sat %': '{:.1f}%',
            'Min/GG': '{:.1f}',
            'Vis/GG': '{:.2f}'
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # --- MAPPA ---
        st.subheader("🗺️ Mappa Territori")
        df_map = df_w.dropna(subset=['latitudine', 'longitudine', 'assigned_rep'])

        if len(df_map) == 0:
            st.warning("⚠️ Nessun cliente valido da visualizzare sulla mappa.")
        else:
            st.caption(f"Visualizzati {len(df_map)} clienti su {len(df_w)} totali")
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
        st.info("👈 Carica il file Excel nella sidebar per iniziare. La simulazione partirà automaticamente.")


if __name__ == "__main__":
    main()
