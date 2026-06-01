import streamlit as st
import streamlit.components.v1 as components
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

def get_alert(s):
    """Restituisce lo stato di saturazione."""
    if s > 110: return "🔴 CRITICO"
    elif s > 100: return "🟠 OVERLOAD"
    elif s > 85: return "⚠️ ATTENZIONE"
    else: return "🟢 OK"

# =============================================================================
# CONFIGURAZIONE
# =============================================================================
PROVINCIAL_CIRCUITY = {
    "MI": 1.25, "LO": 1.20, "CR": 1.20, "MN": 1.25, "BS": 1.25, "BG": 1.25, "PV": 1.20,
    "VC": 1.25, "NO": 1.25, "AL": 1.35, "AT": 1.30, "BI": 1.25, "VB": 1.45, "TO": 1.30,
    "CN": 1.35, "RA": 1.25, "FE": 1.25, "PC": 1.25, "PR": 1.25, "RE": 1.25, "MO": 1.25,
    "BO": 1.30, "RN": 1.25, "FO": 1.25, "FC": 1.30, "PU": 1.30, "AN": 1.30, "MC": 1.30,
    "AP": 1.35, "FM": 1.30, "PE": 1.35, "CH": 1.40, "TE": 1.40, "AQ": 1.55, "CE": 1.35,
    "BN": 1.40, "NA": 1.35, "AV": 1.45, "SA": 1.40, "PZ": 1.50, "MT": 1.45, "BA": 1.30,
    "BR": 1.30, "TA": 1.35, "FG": 1.35, "BT": 1.35, "CB": 1.45, "IS": 1.45, "VT": 1.40,
    "LT": 1.35, "FR": 1.35, "RM": 1.35, "RI": 1.40, "GE": 1.45, "SV": 1.40, "IM": 1.40,
    "SP": 1.40, "LU": 1.35, "PI": 1.30, "PT": 1.30, "PO": 1.30, "LI": 1.30, "AR": 1.35,
    "SI": 1.35, "FI": 1.30, "GR": 1.45, "PG": 1.40, "TR": 1.35, "VI": 1.25, "TV": 1.25,
    "VE": 1.30, "PD": 1.25, "RO": 1.25, "VR": 1.25, "BL": 1.45, "TN": 1.50, "BZ": 1.50,
    "UD": 1.35, "GO": 1.35, "TS": 1.30, "PN": 1.35, "MS": 1.35, "CA": 1.35, "SS": 1.40,
    "NU": 1.45, "OR": 1.40, "OT": 1.45, "SU": 1.45, "RG": 1.45, "SR": 1.40, "CT": 1.35,
    "ME": 1.40, "PA": 1.40, "TP": 1.45, "AG": 1.45, "CL": 1.45, "EN": 1.50, "CS": 1.45,
    "CZ": 1.45, "VV": 1.50, "RC": 1.50, "KR": 1.45, "LE": 1.30, "AO": 1.40, "SO": 1.40,
    "VA": 1.25, "CO": 1.25, "LC": 1.25
}

PROVINCIAL_SPEED = {
    "MI": 38, "RM": 35, "NA": 32, "TO": 40, "GE": 35, "BO": 38, "FI": 36, "VE": 35,
    "BA": 38, "CT": 34, "PA": 36, "LO": 62, "CR": 65, "MN": 60, "PV": 60, "PC": 62,
    "PR": 62, "RE": 62, "MO": 58, "RN": 60, "FE": 60, "RA": 58, "FO": 58, "VR": 60,
    "VI": 60, "PD": 60, "RO": 60, "TV": 60, "BG": 55, "BS": 58, "CO": 50, "VA": 48,
    "NO": 55, "BI": 55, "VC": 55, "AL": 48, "AT": 48, "CN": 50, "LU": 55, "PI": 55,
    "LI": 58, "PO": 55, "PT": 52, "AR": 52, "SI": 52, "GR": 48, "LT": 50, "FR": 52,
    "VT": 50, "RI": 50, "CB": 48, "IS": 48, "CE": 45, "BN": 45, "AV": 45, "SA": 48,
    "PZ": 48, "MT": 48, "FG": 55, "BT": 55, "BR": 58, "TA": 55, "LE": 58, "KR": 52,
    "SV": 48, "IM": 45, "SP": 45, "MS": 50, "PG": 50, "TR": 50, "AN": 52, "MC": 50,
    "AP": 48, "FM": 50, "PE": 48, "CH": 45, "TE": 45, "AQ": 42, "CS": 45, "CZ": 45,
    "VV": 42, "RC": 42, "RG": 48, "SR": 48, "EN": 45, "CL": 45, "AG": 48, "TP": 48,
    "ME": 45, "CA": 48, "SS": 48, "NU": 48, "OR": 48, "OT": 48, "SU": 48, "AO": 42,
    "BL": 45, "BZ": 42, "TN": 42, "SO": 42, "UD": 48, "GO": 48, "PN": 48, "VB": 45,
    "FC": 50, "PU": 48
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
    df['classe'] = np.select(conditions, ['A', 'B', 'C', 'D'], default='D')
    return df

def compute_hull_coords(df_customers):
    if len(df_customers) < 3:
        return None, None
    try:
        pts = df_customers[['longitudine', 'latitudine']].values
        hull = ConvexHull(pts)
        idx = np.append(hull.vertices, hull.vertices[0])
        return pts[idx, 0], pts[idx, 1]
    except:
        return None, None

# =============================================================================
# MODELLO BACINI POLARI + NN PER GIORNATA
# =============================================================================
def _nn_tour_giornata(giornata_df, start_lat, start_lon, circuity_dict, speed_dict):
    """Tour Nearest Neighbor per una singola giornata di lavoro."""
    n = len(giornata_df)
    if n == 0:
        return 0.0, 0.0

    lats = giornata_df['lat'].values
    lons = giornata_df['lon'].values
    siglas = giornata_df['sigla'].values

    remaining = set(range(n))
    tour_km = 0.0
    cur_lat, cur_lon = start_lat, start_lon

    for _ in range(n):
        if not remaining:
            break
        best_idx = None
        best_dist = float('inf')
        for idx in remaining:
            d = haversine_km(cur_lon, cur_lat, lons[idx], lats[idx])
            if d < best_dist:
                best_dist = d
                best_idx = idx

        sigla = siglas[best_idx]
        circ = circuity_dict.get(sigla, 1.35)
        tour_km += best_dist * circ

        remaining.remove(best_idx)
        cur_lat = lats[best_idx]
        cur_lon = lons[best_idx]

    # Ritorno a casa
    return_km = haversine_km(cur_lon, cur_lat, start_lon, start_lat)
    avg_circ = np.mean([circuity_dict.get(s, 1.30) for s in siglas]) if n > 0 else 1.30
    tour_km += return_km * avg_circ

    # Velocità media della giornata
    speeds = [speed_dict.get(s, 45) for s in siglas]
    avg_speed = np.mean(speeds) if speeds else 45

    return tour_km, tour_km / avg_speed


def calculate_travel_km_tours(df_customers, rep_home_lat, rep_home_lon, max_stops_per_day,
                              circuity_dict, speed_dict):
    """
    Modello bacini polari con tour NN per giornata.
    """
    if len(df_customers) == 0:
        return 0.0, 0.0

    df = df_customers.copy()
    total_visits = df['freq_visite'].sum()
    if total_visits == 0:
        return 0.0, 0.0

    n_customers = len(df)
    n_sectors = min(12, max(4, int(np.ceil(np.sqrt(n_customers)))))

    dx = np.radians(df['longitudine'].values - rep_home_lon)
    dy = np.radians(df['latitudine'].values - rep_home_lat)
    angles = np.degrees(np.arctan2(dx, dy))
    angles = (angles + 360) % 360
    df['angle'] = angles
    df['sector'] = (angles / (360 / n_sectors)).astype(int) % n_sectors

    visits = []
    for idx, row in df.iterrows():
        freq = int(row['freq_visite']) if pd.notna(row['freq_visite']) else 0
        for _ in range(freq):
            visits.append({
                'cliente_idx': idx,
                'sector': int(row['sector']),
                'lat': row['latitudine'],
                'lon': row['longitudine'],
                'sigla': row['sigla'],
                'dist': row.get('dist_km',
                    haversine_km(row['longitudine'], row['latitudine'], rep_home_lon, rep_home_lat))
            })

    if len(visits) == 0:
        return 0.0, 0.0

    visits_df = pd.DataFrame(visits)
    n_visits = len(visits_df)
    visit_mask = np.zeros(n_visits, dtype=bool)

    giornate = []

    while not visit_mask.all():
        residue = {}
        for s in range(n_sectors):
            residue[s] = ((visits_df['sector'] == s) & (~visit_mask)).sum()

        best_sector = max(residue, key=residue.get)
        if residue[best_sector] == 0:
            break

        avail_best = visits_df.loc[~visit_mask & (visits_df['sector'] == best_sector)]
        avail_best = avail_best.sort_values('dist')
        take_best = avail_best.head(max_stops_per_day)
        taken_indices = take_best.index.tolist()
        visit_mask[taken_indices] = True

        remaining_slots = max_stops_per_day - len(taken_indices)
        if remaining_slots > 0:
            for adj in [(best_sector - 1) % n_sectors, (best_sector + 1) % n_sectors]:
                if remaining_slots <= 0:
                    break
                avail_adj = visits_df.loc[~visit_mask & (visits_df['sector'] == adj)]
                avail_adj = avail_adj.sort_values('dist')
                take_adj = avail_adj.head(remaining_slots)
                adj_indices = take_adj.index.tolist()
                if adj_indices:
                    taken_indices.extend(adj_indices)
                    visit_mask[adj_indices] = True
                    remaining_slots -= len(adj_indices)

        if len(taken_indices) > 0:
            giornata_df = visits_df.loc[taken_indices]
            km_giorno, ore_giorno = _nn_tour_giornata(
                giornata_df, rep_home_lat, rep_home_lon,
                circuity_dict, speed_dict
            )
            giornate.append({
                'sector': best_sector,
                'n_visite': len(taken_indices),
                'km': km_giorno,
                'ore_viaggio': ore_giorno
            })

    total_km = sum(g['km'] for g in giornate)
    total_ore = sum(g['ore_viaggio'] for g in giornate)

    return total_km, total_ore


def run_simulation(df_c, df_v, active_list, col_vol, da_a, da_b, da_c,
                   freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro,
                   max_stops_per_day, use_nearest_neighbor=False):
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

    # Assegnazione venditori (su tutto il df, prima della classificazione)
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

    # Classificazione ABC su tutti i clienti assegnati
    df_w = classify_abc(df_w, col_vol, da_a, da_b, da_c)

    # Aggregati completi per venditore (inclusi D)
    agg_list = []
    for rep in active_list:
        sub = df_w[df_w['assigned_rep'] == rep]
        if len(sub) > 0:
            agg_list.append({
                'sales_rep': rep,
                'n_clienti': int(len(sub)),
                'n_classe_a': int((sub['classe'] == 'A').sum()),
                'n_classe_b': int((sub['classe'] == 'B').sum()),
                'n_classe_c': int((sub['classe'] == 'C').sum()),
                'n_classe_d': int((sub['classe'] == 'D').sum()),
                'volume_abc': float(sub.loc[sub['classe'].isin(['A','B','C']), col_vol].sum()),
                'volume_d': float(sub.loc[sub['classe'] == 'D', col_vol].sum()),
            })
        else:
            agg_list.append({
                'sales_rep': rep, 'n_clienti': 0,
                'n_classe_a': 0, 'n_classe_b': 0, 'n_classe_c': 0, 'n_classe_d': 0,
                'volume_abc': 0.0, 'volume_d': 0.0
            })
    agg = pd.DataFrame(agg_list)

    # Filtra clienti attivi (A,B,C) per la logica visite e km
    df_work = df_w[df_w['classe'] != 'D'].copy()
    if len(df_work) == 0:
        return None, "Nessun cliente attivo sopra la soglia minima C."

    # Coordinate venditore su df_work
    df_work['rep_lat'] = df_work['assigned_rep'].map(df_v_valid.set_index('sales rep')['latitudine'])
    df_work['rep_lon'] = df_work['assigned_rep'].map(df_v_valid.set_index('sales rep')['longitudine'])
    if 'sigla' in df_v_valid.columns:
        df_work['rep_sigla'] = df_work['assigned_rep'].map(df_v_valid.set_index('sales rep')['sigla'])
    else:
        df_work['rep_sigla'] = None

    freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c}
    df_work['freq_visite'] = df_work['classe'].map(freq_map)
    df_work['ore_visita_annue'] = (df_work['freq_visite'] * dur_visita) / 60.0

    # Calcolo km e ore viaggio
    travel_data = []
    for rep in active_list:
        sub = df_work[df_work['assigned_rep'] == rep]
        if len(sub) > 0:
            rep_lat = sub['rep_lat'].iloc[0]
            rep_lon = sub['rep_lon'].iloc[0]
            km_totali, ore_viag = calculate_travel_km_tours(
                sub, rep_lat, rep_lon, max_stops_per_day,
                PROVINCIAL_CIRCUITY, PROVINCIAL_SPEED
            )
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': ore_viag, 'km_annui': km_totali})
        else:
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': 0.0, 'km_annui': 0.0})

    travel_df = pd.DataFrame(travel_data)
    agg = agg.merge(travel_df, on='sales_rep', how='left').fillna(0)

    # Ore visite aggregate
    visite_agg = df_work.groupby('assigned_rep').agg(
        ore_visite_annue=('ore_visita_annue', 'sum')
    ).reset_index().rename(columns={'assigned_rep': 'sales_rep'})
    agg = agg.merge(visite_agg, on='sales_rep', how='left').fillna(0)

    max_ore_campo = ore_gg * gg_lavoro
    agg['ore_totali_annue'] = agg['ore_visite_annue'] + agg['ore_viaggio_annue']
    agg['saturazione_pct'] = np.where(max_ore_campo > 0, (agg['ore_totali_annue'] / max_ore_campo) * 100, 0.0)
    agg['driving_min_giorno'] = (agg['ore_viaggio_annue'] * 60) / gg_lavoro
    agg['visite_giorno'] = (agg['ore_visite_annue'] * 60 / dur_visita) / gg_lavoro
    agg['stato'] = agg['saturazione_pct'].apply(get_alert)

    all_reps = pd.DataFrame({'sales_rep': active_list})
    result = all_reps.merge(agg, on='sales_rep', how='left').fillna(0)
    result = result.sort_values('saturazione_pct', ascending=False)
    return result, df_work

# =============================================================================
# INTERFACCIA
# =============================================================================

# =============================================================================
# FUNZIONI DOWNSIZE — RIASSENZIONE CLIENTI
# =============================================================================

def compute_neighbor_reps(removed_rep, df_c, df_v, active_reps, top_n=8):
    """
    Trova i venditori limitrofi al venditore rimosso.
    Score spaziale = 0.4*(1/distanza_media) + 0.4*(clienti_per_cui_è_NN) + 0.2*(1/distanza_minima)
    """
    clients_removed = df_c[df_c['sales rep'] == removed_rep]
    if len(clients_removed) == 0:
        return []

    c_lats = clients_removed['latitudine'].values
    c_lons = clients_removed['longitudine'].values
    n_cli = len(clients_removed)

    scores = []
    for rep in active_reps:
        if rep == removed_rep:
            continue
        v_row = df_v[df_v['sales rep'] == rep]
        if len(v_row) == 0:
            continue
        v_lat = v_row['latitudine'].iloc[0]
        v_lon = v_row['longitudine'].iloc[0]
        if pd.isna(v_lat) or pd.isna(v_lon):
            continue

        dists = haversine_km(c_lons, c_lats, v_lon, v_lat)
        avg_dist = float(np.mean(dists))
        min_dist = float(np.min(dists))

        # Quanti clienti del rimosso hanno 'rep' come nearest neighbor?
        nn_count = 0
        for _, row in clients_removed.iterrows():
            best_dist = float('inf')
            best_rep = None
            for r2 in active_reps:
                if r2 == removed_rep:
                    continue
                v2 = df_v[df_v['sales rep'] == r2]
                if len(v2) > 0 and pd.notna(v2['latitudine'].iloc[0]):
                    d2 = haversine_km(row['longitudine'], row['latitudine'],
                                      v2['longitudine'].iloc[0], v2['latitudine'].iloc[0])
                    if d2 < best_dist:
                        best_dist = d2
                        best_rep = r2
            if best_rep == rep:
                nn_count += 1

        spatial_score = (0.4 * (1.0 / (avg_dist + 1.0)) + 
                        0.4 * (nn_count / max(n_cli, 1)) + 
                        0.2 * (1.0 / (min_dist + 1.0)))
        scores.append({
            'rep': rep,
            'score': spatial_score,
            'avg_dist_km': round(avg_dist, 1),
            'min_dist_km': round(min_dist, 1),
            'nn_count': int(nn_count),
            'home_lat': float(v_lat),
            'home_lon': float(v_lon),
            'n_clients': n_cli
        })

    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[:top_n]


def auto_reassign_greedy(removed_rep, selected_receivers, df_c, df_v, current_result,
                         dur_visita, freq_a, freq_b, freq_c, ore_gg_eff, gg_lavoro,
                         col_vol, lambda_sat=1.5, mu_overload=3.0):
    """
    Algoritmo GSSH (Greedy Spatial-Saturation Heuristic).
    Riassegna clienti del venditore rimosso privilegiando distanza e saturazione residua.
    """
    removed_mask = df_c['sales rep'] == removed_rep
    removed_clients = df_c[removed_mask].copy()

    if len(removed_clients) == 0:
        return df_c.copy(), []

    # Capacità e ore residue
    cap = ore_gg_eff * gg_lavoro
    residual_hours = {}
    current_sat = {}
    for _, row in current_result.iterrows():
        rep = row['sales_rep']
        used = row['ore_totali_annue']
        residual_hours[rep] = max(0.0, cap - used)
        current_sat[rep] = row['saturazione_pct']

    # Coordinate riceventi (home base)
    rec_coords = {}
    for rep in selected_receivers:
        v = df_v[df_v['sales rep'] == rep]
        if len(v) > 0 and pd.notna(v['latitudine'].iloc[0]):
            rec_coords[rep] = (float(v['latitudine'].iloc[0]), float(v['longitudine'].iloc[0]))
        else:
            sub = df_c[df_c['sales rep'] == rep]
            if len(sub) > 0:
                rec_coords[rep] = (float(sub['latitudine'].mean()), float(sub['longitudine'].mean()))

    if not rec_coords:
        return df_c.copy(), []

    # Assicurati che ci sia la colonna classe
    freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c, 'D': 0}
    if 'classe' not in removed_clients.columns:
        removed_clients['classe'] = 'C'

    # Ordina: A prima, poi B, poi C, poi D; all'interno per volume decrescente
    classe_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    removed_clients['_sort_key'] = removed_clients['classe'].map(classe_order)
    if col_vol and col_vol in removed_clients.columns:
        removed_clients[col_vol] = pd.to_numeric(removed_clients[col_vol], errors='coerce').fillna(0)
        removed_clients = removed_clients.sort_values(['_sort_key', col_vol], ascending=[True, False])
    else:
        removed_clients = removed_clients.sort_values('_sort_key')

    # Stima carico incrementale in ore/anno per un cliente
    def est_load(row):
        cls = row.get('classe', 'C')
        if cls == 'D':
            return 0.0
        freq = freq_map.get(cls, 12)
        ore_vis = (freq * dur_visita) / 60.0
        # Proxy viaggio: +30% sulle ore visita per viaggio semplificato
        # (dopo il commit verrà ricalcolato esattamente con il tour NN)
        return ore_vis * 1.35

    new_assignments = []
    residual_dynamic = residual_hours.copy()

    for idx, row in removed_clients.iterrows():
        c_lat = row['latitudine']
        c_lon = row['longitudine']
        if pd.isna(c_lat) or pd.isna(c_lon):
            continue

        load = est_load(row)
        best_rep = None
        best_score = float('inf')

        for rep in selected_receivers:
            if rep not in rec_coords:
                continue
            r_lat, r_lon = rec_coords[rep]
            dist = haversine_km(c_lon, c_lat, r_lon, r_lat)

            res = residual_dynamic.get(rep, 0.0)
            sat_pct = 100.0 * (1.0 - res / cap) if cap > 0 else 0.0

            penalty = 1.0
            if res > 0:
                penalty += lambda_sat * (sat_pct / 100.0)
            else:
                penalty += mu_overload  # saturo!

            if sat_pct > 100.0:
                penalty += mu_overload * ((sat_pct - 100.0) / 50.0)

            score = dist * penalty
            if score < best_score:
                best_score = score
                best_rep = rep

        if best_rep:
            new_assignments.append((idx, best_rep, load))
            residual_dynamic[best_rep] = max(0.0, residual_dynamic.get(best_rep, 0.0) - load)

    # Applica assegnazioni
    df_c_new = df_c.copy()
    for idx, rep, _ in new_assignments:
        df_c_new.at[idx, 'sales rep'] = rep
        if 'assigned_rep' in df_c_new.columns:
            df_c_new.at[idx, 'assigned_rep'] = rep

    return df_c_new, new_assignments


def get_saturation_badge(sat_pct):
    if sat_pct > 110:
        return "🔴 CRITICO", "#ffcdd2", "#b71c1c"
    elif sat_pct > 100:
        return "🟠 OVERLOAD", "#ffe0b2", "#e65100"
    elif sat_pct > 85:
        return "⚠️ ATTENZIONE", "#fff9c4", "#f57f17"
    else:
        return "🟢 OK", "#c8e6c9", "#1b5e20"


def main():
    # =============================================================================
    # INIZIALIZZAZIONE STATO GLOBALE
    # =============================================================================
    if 'abc_vals' not in st.session_state or st.session_state.abc_vals is None:
        st.session_state.abc_vals = {
            'da_a': 801, 'a_a': -1, 'freq_a': 24,
            'da_b': 401, 'a_b': 800, 'freq_b': 16,
            'da_c': 101, 'a_c': 400, 'freq_c': 12,
            'da_d': 0, 'a_d': 100, 'freq_d': 0
        }
    if 'abc_version' not in st.session_state:
        st.session_state.abc_version = 2

    # Ripristino venditore da annullamento downsize
    if 'restore_rep' in st.session_state:
        r = st.session_state.restore_rep
        st.session_state[f"rep_{r}"] = True
        del st.session_state.restore_rep

    # Scroll in alto quando entra in downsize
    if st.session_state.get('downsize_pending', False):
        components.html("<script>window.parent.scrollTo({top:0,behavior:'smooth'});setTimeout(function(){var el=document.querySelector('[data-testid=stAppViewContainer]');if(el)el.scrollTop=0;},100);</script>", height=0)

    st.markdown(""""
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

    # =============================================================================
    # SIDEBAR
    # =============================================================================
    with st.sidebar:
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        manual_run = st.button("🚀 LANCIA SIMULAZIONE", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

        st.subheader("📁 Dati di Input")
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
        pausa_pranzo = st.slider("🍽️ Pausa pranzo (min/giorno)", 0, 120, 100, step=5)
        ore_effettive_gg = ore_gg - (pausa_pranzo / 60.0)
        st.caption(f"*Capacità annua: {ore_effettive_gg * gg_lavoro:,.0f} ore*")
        max_stops_per_day = st.slider("📦 Max visite/giorno", 3, 10, 5, step=1)
        st.subheader("👥 Stato Venditori")
        reps = sorted(df_v['sales rep'].unique())
        # =============================================================================
        # STATO DOWNSIZE — sanitizzazione
        # =============================================================================
        is_pending = st.session_state.get('downsize_pending', False)
        removed_rep_pending = st.session_state.get('removed_rep', None)

        # Se lo stato è inconsistente, pulisci
        if is_pending and (removed_rep_pending not in reps or 'pre_downsize_df_c' not in st.session_state):
            st.session_state.downsize_pending = False
            st.session_state.removed_rep = None
            st.session_state.downsize_candidates = []
            st.session_state.downsize_selected_receivers = []
            st.session_state.downsize_preview_df = None
            st.session_state.downsize_assignments = []
            st.session_state.downsize_preview_result = None
            st.session_state.downsize_preview_df_work = None
            is_pending = False

        # Inizializza active_snapshot se nuovo file o mancante
        if 'file_signature' not in st.session_state or st.session_state.file_signature != file_signature:
            st.session_state.file_signature = file_signature
            st.session_state.active_snapshot = set(reps)
            st.session_state.downsize_pending = False
            st.session_state.removed_rep = None
            st.session_state.downsize_candidates = []
            st.session_state.downsize_selected_receivers = []
            st.session_state.downsize_preview_df = None
            st.session_state.downsize_assignments = []
            st.session_state.downsize_preview_result = None
            st.session_state.downsize_preview_df_work = None

        # =============================================================================
        # SIDEBAR: Venditori Attivi/Disattivi
        # =============================================================================
        with st.expander("Attiva / Disattiva venditori", expanded=True):
            if is_pending:
                # Durante downsize: stato statico, nessuna checkbox
                for r in reps:
                    if r == removed_rep_pending:
                        st.markdown(f"❌ ~~{r}~~ *(in riassegnazione)*")
                    else:
                        st.markdown(f"✅ {r}")
                rep_status = {r: (r != removed_rep_pending) for r in reps}
            else:
                rep_status = {}
                for r in reps:
                    val = r in st.session_state.active_snapshot
                    rep_status[r] = st.checkbox(r, value=val, key=f"rep_{r}")

        # =============================================================================
        # TRIGGER DOWNSIZE (solo se non in pending)
        # =============================================================================
        if not is_pending:
            active_list_sidebar = [r for r, s in rep_status.items() if s]
            removed = st.session_state.active_snapshot - set(active_list_sidebar)
            if removed:
                removed_rep = sorted(list(removed))[0]
                st.session_state.downsize_pending = True
                st.session_state.removed_rep = removed_rep
                st.session_state.pre_downsize_df_c = df_c.copy()
                st.session_state.pre_downsize_active = list(st.session_state.active_snapshot)
                st.session_state.downsize_candidates = []
                st.session_state.downsize_selected_receivers = []
                st.session_state.downsize_preview_df = None
                st.session_state.downsize_assignments = []
                st.session_state.downsize_preview_result = None
                st.session_state.downsize_preview_df_work = None
                st.rerun()
            else:
                st.session_state.active_snapshot = set(active_list_sidebar)

    # =============================================================================
    # PANNELLO DOWNSIZE INTERATTIVO
    # =============================================================================
    if st.session_state.get('downsize_pending', False) and uploaded:
        removed_rep = st.session_state.removed_rep
        pre_df = st.session_state.pre_downsize_df_c
        n_removed = len(pre_df[pre_df['sales rep'] == removed_rep])

        st.markdown("""
        <style>
        .downsize-box {
            background: linear-gradient(135deg, #2d132c 0%, #1a1a2e 100%);
            border: 2px solid #e94560;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(233, 69, 96, 0.15);
        }
        .downsize-title { color: #e94560; font-size: 28px; font-weight: 700; margin: 0 0 8px 0; }
        .downsize-sub  { color: #ccc; font-size: 16px; margin: 0; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="downsize-box">
            <div class="downsize-title">⚠️ DOWNSIZE IN CORSO</div>
            <div class="downsize-sub">
                Venditore rimosso: <b>{removed_rep}</b> &nbsp;|&nbsp;
                Clienti da riassegnare: <b>{n_removed}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Calcola candidati limitrofi
        if not st.session_state.get('downsize_candidates', []):
            active_for_neighbor = [r for r in reps if r != removed_rep]
            candidates = compute_neighbor_reps(
                removed_rep, pre_df, df_v, active_for_neighbor, top_n=8
            )
            st.session_state.downsize_candidates = candidates

        candidates = st.session_state.downsize_candidates

        if not candidates:
            st.error("❌ Nessun venditore limitrofo trovato con coordinate valide.")
            if st.button("🔙 Chiudi e Ripristina", use_container_width=True):
                st.session_state.downsize_pending = False
                st.session_state.removed_rep = None
                st.session_state.downsize_candidates = []
                st.session_state.active_snapshot = set(st.session_state.pre_downsize_active)
                st.rerun()
        else:
            st.subheader("👥 1. Seleziona i venditori riceventi")
            st.caption("Scegli uno o più venditori che riceveranno i clienti. Ordinati per prossimità territoriale.")

            abc_vals = st.session_state.get('abc_vals', {
                'freq_a': 24, 'freq_b': 16, 'freq_c': 12
            })
            fa = abc_vals.get('freq_a', 24)
            fb = abc_vals.get('freq_b', 16)
            fc = abc_vals.get('freq_c', 12)
            da_a = abc_vals.get('da_a', 801)
            da_b = abc_vals.get('da_b', 401)
            da_c = abc_vals.get('da_c', 101)

            n_cols = min(len(candidates), 3)
            cols = st.columns(n_cols)
            selected_receivers = []

            for i, cand in enumerate(candidates):
                rep = cand['rep']
                with cols[i % n_cols]:
                    sat_pre = 0.0
                    sat_label = "N/D"
                    if st.session_state.current_result is not None:
                        sat_row = st.session_state.current_result[
                            st.session_state.current_result['sales_rep'] == rep
                        ]
                        if len(sat_row) > 0:
                            sat_pre = sat_row['saturazione_pct'].iloc[0]
                            sat_label = f"{sat_pre:.1f}%"

                    badge, bg, fg = get_saturation_badge(sat_pre)
                    label_text = f"**{rep}**\n\n{badge} Sat: {sat_label}\n\n📍 Distanza media: **{cand['avg_dist_km']} km**\n\n🎯 Clienti vicini: **{cand['nn_count']}** / {cand['n_clients']}"
                    chk = st.checkbox(label_text, key=f"recv_{rep}")
                    if chk:
                        selected_receivers.append(rep)

            st.session_state.downsize_selected_receivers = selected_receivers

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

            with btn_col1:
                disable_auto = len(selected_receivers) == 0 or st.session_state.current_result is None
                if st.session_state.current_result is None:
                    st.caption("⚠️ Lancia prima una simulazione per avere i dati di saturazione.")

                if st.button("🤖 Auto-Alloca Clienti", disabled=disable_auto, use_container_width=True, type="primary"):
                    df_new, assigns = auto_reassign_greedy(
                        removed_rep,
                        selected_receivers,
                        st.session_state.pre_downsize_df_c,
                        df_v,
                        st.session_state.current_result,
                        dur_visita,
                        fa, fb, fc,
                        ore_effettive_gg,
                        gg_lavoro,
                        st.session_state.get('col_vol'),
                        da_a, da_b, da_c
                    )
                    # Calcolo VERO con run_simulation
                    new_active = [r for r in st.session_state.pre_downsize_active if r != removed_rep]
                    abc = st.session_state.get('abc_vals', {
                'da_a': 801, 'a_a': -1, 'freq_a': 24,
                'da_b': 401, 'a_b': 800, 'freq_b': 16,
                'da_c': 101, 'a_c': 400, 'freq_c': 12,
                'da_d': 0, 'a_d': 100, 'freq_d': 0
            })
                    preview_res, preview_df_w = run_simulation(
                        df_new, df_v, new_active, col_vol,
                        abc['da_a'], abc['da_b'], abc['da_c'],
                        abc['freq_a'], abc['freq_b'], abc['freq_c'],
                        dur_visita, ore_effettive_gg, gg_lavoro,
                        max_stops_per_day, use_nearest_neighbor=False
                    )
                    if preview_res is None:
                        st.error(f"❌ Errore calcolo preview: {preview_df_w}")
                    else:
                        st.session_state.downsize_preview_df = df_new
                        st.session_state.downsize_assignments = assigns
                        st.session_state.downsize_preview_result = preview_res
                        st.session_state.downsize_preview_df_work = preview_df_w
                        st.rerun()

            with btn_col2:
                disable_confirm = st.session_state.downsize_preview_result is None
                if st.button("✅ Conferma e Applica", disabled=disable_confirm, use_container_width=True):
                    # Salva scenario pre-downsize per confronto
                    if st.session_state.current_result is not None:
                        st.session_state.pre_downsize_result = st.session_state.current_result.copy()
                        st.session_state.pre_downsize_df_work = st.session_state.current_df_work.copy() if st.session_state.current_df_work is not None else None
                        st.session_state.pre_downsize_params = st.session_state.current_params.copy() if st.session_state.current_params else {}

                    # Applica modifiche
                    st.session_state.df_c = st.session_state.downsize_preview_df.copy()
                    st.session_state.current_result = st.session_state.downsize_preview_result.copy()
                    st.session_state.current_df_work = st.session_state.downsize_preview_df_work.copy() if st.session_state.downsize_preview_df_work is not None else None

                    # Aggiorna active_snapshot (rimuovi definitivamente)
                    new_active = [r for r in st.session_state.pre_downsize_active if r != removed_rep]
                    st.session_state.active_snapshot = set(new_active)

                    # Pulisci stato downsize
                    st.session_state.downsize_pending = False
                    st.session_state.removed_rep = None
                    st.session_state.downsize_candidates = []
                    st.session_state.downsize_selected_receivers = []
                    st.session_state.downsize_preview_df = None
                    st.session_state.downsize_assignments = []
                    st.session_state.downsize_preview_result = None
                    st.session_state.downsize_preview_df_work = None

                    # Aggiorna params
                    st.session_state.current_params = {
                        'active_list': new_active,
                        'ore_gg': ore_effettive_gg,
                        'gg_lavoro': gg_lavoro,
                        'dur_visita': dur_visita,
                        'max_stops': max_stops_per_day,
                        'reps': reps,
                        'use_nn': False,
                        'modo': "📍 Mappa Attuale (Assegnazione Reale)",
                        'min_vol': min_vol
                    }

                    st.success("✅ Downsize applicato! Confronto Prima/Dopo disponibile sotto.")
                    st.rerun()

            with btn_col3:
                if st.button("❌ Annulla e Ripristina", use_container_width=True):
                    st.session_state.downsize_pending = False
                    st.session_state.removed_rep = None
                    st.session_state.downsize_candidates = []
                    st.session_state.downsize_selected_receivers = []
                    st.session_state.downsize_preview_df = None
                    st.session_state.downsize_assignments = []
                    st.session_state.downsize_preview_result = None
                    st.session_state.downsize_preview_df_work = None
                    st.session_state.active_snapshot = set(st.session_state.pre_downsize_active)
                    st.rerun()

            # =============================================================================
            # PREVIEW RIASSENZIONE (DATI VERO MOTORE)
            # =============================================================================
            if st.session_state.downsize_preview_result is not None:
                st.divider()
                st.subheader("📊 2. Preview Riassegnazione — Dati Veri")

                preview_res = st.session_state.downsize_preview_result
                pre_res = st.session_state.current_result

                preview_rows = []
                for rep in selected_receivers:
                    pre_row = pre_res[pre_res['sales_rep'] == rep]
                    post_row = preview_res[preview_res['sales_rep'] == rep]

                    sat_pre = pre_row['saturazione_pct'].iloc[0] if len(pre_row) > 0 else 0.0
                    sat_post = post_row['saturazione_pct'].iloc[0] if len(post_row) > 0 else 0.0
                    cli_pre = int(pre_row['n_clienti'].iloc[0]) if len(pre_row) > 0 else 0
                    cli_post = int(post_row['n_clienti'].iloc[0]) if len(post_row) > 0 else 0
                    km_pre = pre_row['km_annui'].iloc[0] if len(pre_row) > 0 else 0.0
                    km_post = post_row['km_annui'].iloc[0] if len(post_row) > 0 else 0.0
                    ore_pre = pre_row['ore_totali_annue'].iloc[0] if len(pre_row) > 0 else 0.0
                    ore_post = post_row['ore_totali_annue'].iloc[0] if len(post_row) > 0 else 0.0

                    preview_rows.append({
                        'Ricevente': rep,
                        'Clienti': f"{cli_pre} → {cli_post} ({cli_post - cli_pre:+d})",
                        'Sat %': f"{sat_pre:.1f}% → {sat_post:.1f}%",
                        'Δ Sat': f"{sat_post - sat_pre:+.1f}%",
                        'Ore Totali': f"{ore_pre:.1f} → {ore_post:.1f}h",
                        'Km Annui': f"{km_pre:,.0f} → {km_post:,.0f} km"
                    })

                if preview_rows:
                    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

                # Mappa preview
                st.subheader("🗺️ 3. Mappa Preview Spostamenti")
                df_prev = st.session_state.downsize_preview_df
                df_pre = st.session_state.pre_downsize_df_c
                moved_mask = (df_prev['sales rep'] != df_pre['sales rep']) & (df_pre['sales rep'] == removed_rep)
                df_moved = df_prev[moved_mask].copy()

                if len(df_moved) > 0:
                    fig_d = go.Figure()
                    colors_d = px.colors.qualitative.Bold
                    rec_color_map = {rep: colors_d[i % len(colors_d)] for i, rep in enumerate(selected_receivers)}

                    for rep in selected_receivers:
                        sub = df_moved[df_moved['sales rep'] == rep]
                        if len(sub) > 0:
                            fig_d.add_trace(go.Scattermapbox(
                                lat=sub['latitudine'], lon=sub['longitudine'],
                                mode='markers',
                                marker=dict(size=10, color=rec_color_map[rep], opacity=0.9),
                                name=f"→ {rep} ({len(sub)} clienti)",
                                hoverinfo='name'
                            ))
                            v_home = df_v[df_v['sales rep'] == rep]
                            if len(v_home) > 0 and pd.notna(v_home['latitudine'].iloc[0]):
                                for _, row in sub.iterrows():
                                    fig_d.add_trace(go.Scattermapbox(
                                        mode='lines',
                                        lat=[row['latitudine'], v_home['latitudine'].iloc[0]],
                                        lon=[row['longitudine'], v_home['longitudine'].iloc[0]],
                                        line=dict(width=1, color=rec_color_map[rep]),
                                        opacity=0.4,
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))

                    df_v_recv = df_v[df_v['sales rep'].isin(selected_receivers)].dropna(subset=['latitudine', 'longitudine'])
                    if len(df_v_recv) > 0:
                        fig_d.add_trace(go.Scattermapbox(
                            lat=df_v_recv['latitudine'], lon=df_v_recv['longitudine'],
                            mode='markers',
                            marker=dict(size=16, color='white', opacity=0.9, symbol='star'),
                            name='⭐ Home Base Riceventi',
                            hoverinfo='name'
                        ))

                    v_removed = df_v[df_v['sales rep'] == removed_rep]
                    if len(v_removed) > 0 and pd.notna(v_removed['latitudine'].iloc[0]):
                        fig_d.add_trace(go.Scattermapbox(
                            lat=v_removed['latitudine'], lon=v_removed['longitudine'],
                            mode='markers',
                            marker=dict(size=18, color='gray', opacity=0.5, symbol='x'),
                            name=f'❌ {removed_rep} (rimosso)',
                            hoverinfo='name'
                        ))

                    fig_d.update_layout(
                        mapbox_style="carto-positron",
                        mapbox_zoom=6,
                        mapbox_center=dict(
                            lat=df_moved['latitudine'].mean() if len(df_moved) > 0 else 42.5,
                            lon=df_moved['longitudine'].mean() if len(df_moved) > 0 else 12.5
                        ),
                        margin=dict(r=0, t=30, l=0, b=0),
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                                   bgcolor='rgba(255,255,255,0.9)')
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                else:
                    st.info("Nessun cliente da visualizzare sulla mappa.")

                # Alert clienti lontani
                distant = []
                for _, row in df_moved.iterrows():
                    new_rep = row['sales rep']
                    v_new = df_v[df_v['sales rep'] == new_rep]
                    if len(v_new) > 0:
                        d = haversine_km(row['longitudine'], row['latitudine'],
                                         v_new['longitudine'].iloc[0], v_new['latitudine'].iloc[0])
                        if d > 50:
                            distant.append(f"{row.get('ragione sociale','Cliente')} → {new_rep} ({d:.1f} km)")

                if distant:
                    with st.expander(f"⚠️ {len(distant)} clienti spostati oltre 50 km — Review consigliata"):
                        for item in distant[:10]:
                            st.write(f"• {item}")
                        if len(distant) > 10:
                            st.write(f"... e altri {len(distant)-10}")

        st.divider()
        st.stop()

        st.session_state.df_c = df_c
        st.session_state.df_c = df_c
        st.session_state.col_vol = col_vol
        st.session_state.col_vol = col_vol

        st.divider()
        st.subheader("📊 Matrice Classificazione ABC & Frequenze")
        st.caption("Definisci gli intervalli esatti (da/a) e le visite annue per ogni classe.")

        # NUOVI DEFAULT: D=0-100, C=101-400, B=401-800, A=801+
        # Forza reset matrice se versione cambiata
        if 'abc_version' not in st.session_state:
            st.session_state.abc_version = 2
            # abc_vals già inizializzato all'inizio di main()

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
            st.markdown("**⚫ Classe D**")
            da_d = st.number_input("da ≥", key="da_d_input", value=st.session_state.abc_vals['da_d'], min_value=0, step=1)
            a_d = st.number_input("a <", key="a_d_input", value=st.session_state.abc_vals['a_d'], min_value=-1, step=1, help="-1 = infinito")
            freq_d = st.number_input("Visite/anno", key="freq_d_input", value=st.session_state.abc_vals['freq_d'], min_value=0, step=1)
        st.session_state.abc_vals = {
            'da_a': da_a, 'a_a': a_a, 'freq_a': freq_a,
            'da_b': da_b, 'a_b': a_b, 'freq_b': freq_b,
            'da_c': da_c, 'a_c': a_c, 'freq_c': freq_c,
            'da_d': da_d, 'a_d': a_d, 'freq_d': freq_d
        }
        min_vol = da_c

        # Preview distribuzione
        try:
            df_preview = df_c.copy()
            df_preview[col_vol] = pd.to_numeric(df_preview[col_vol], errors='coerce').fillna(0)
            df_preview = classify_abc(df_preview, col_vol, da_a, da_b, da_c)
            dist = df_preview['classe'].value_counts()
            col_prev1, col_prev2, col_prev3, col_prev4 = st.columns(4)
            with col_prev1: st.metric("🟢 Classe A", f"{fmt_eu(dist.get('A', 0))}")
            with col_prev2: st.metric("🟡 Classe B", f"{fmt_eu(dist.get('B', 0))}")
            with col_prev3: st.metric("🔴 Classe C", f"{fmt_eu(dist.get('C', 0))}")
            with col_prev4: st.metric("⚫ Classe D", f"{fmt_eu(dist.get('D', 0))}")
        except: pass

        if st.button("🔄 Aggiorna Classificazione", use_container_width=True):
            st.session_state.abc_updated = True
            st.rerun()

    # =============================================================================
    # LOGICA ESECUZIONE
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

    if run_sim and uploaded and not st.session_state.get('downsize_pending', False):
        with st.spinner("🔄 Calcolo scenario Density-Aware in corso..."):
            try:
                active_list = [r for r, s in rep_status.items() if s]
                if len(active_list) == 0:
                    st.error("⚠️ Seleziona almeno un venditore")
                else:
                    abc = st.session_state.get('abc_vals', {
                'da_a': 801, 'a_a': -1, 'freq_a': 24,
                'da_b': 401, 'a_b': 800, 'freq_b': 16,
                'da_c': 101, 'a_c': 400, 'freq_c': 12,
                'da_d': 0, 'a_d': 100, 'freq_d': 0
            })
                    res, df_w = run_simulation(df_c, df_v, active_list, col_vol, 
                                               abc['da_a'], abc['da_b'], abc['da_c'],
                                               abc['freq_a'], abc['freq_b'], abc['freq_c'],
                                               dur_visita, ore_effettive_gg, gg_lavoro,
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

        with st.expander("ℹ️ Cosa significano le colonne?", expanded=False):
            st.markdown("""
            - **Sat %** → Saturazione annua = Ore Totali / (Ore/Giorno × Giorni Lavoro)
            - **Min/GG** → Minuti medi di guida al giorno = Ore Viaggio Annue × 60 / Giorni Lavoro
            - **Vis/GG** → Visite medie al giorno = Ore Visite Annue × 60 / Durata Visita / Giorni Lavoro
            """)

        disp = res[['sales_rep', 'stato', 'n_clienti', 'n_classe_a', 'n_classe_b', 'n_classe_c', 'n_classe_d',
                    'volume_abc', 'volume_d', 'ore_visite_annue', 'ore_viaggio_annue', 'ore_totali_annue',
                    'saturazione_pct', 'driving_min_giorno', 'visite_giorno']].copy()
        disp.columns = ['Venditore', 'Stato', 'Clienti', 'A', 'B', 'C', 'D', 'Volumi A-B-C', 'Volumi D',
                        'Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Sat %', 'Min/GG', 'Vis/GG']

        # Riga TOTALE
        total_row = pd.DataFrame([{
            'Venditore': 'TOTALE',
            'Stato': get_alert(disp['Sat %'].mean()),
            'Clienti': disp['Clienti'].sum(),
            'A': disp['A'].sum(),
            'B': disp['B'].sum(),
            'C': disp['C'].sum(),
            'D': disp['D'].sum(),
            'Volumi A-B-C': disp['Volumi A-B-C'].sum(),
            'Volumi D': disp['Volumi D'].sum(),
            'Ore Visite': disp['Ore Visite'].sum(),
            'Ore Viaggio': disp['Ore Viaggio'].sum(),
            'Ore Totali': disp['Ore Totali'].sum(),
            'Sat %': disp['Sat %'].mean(),
            'Min/GG': disp['Min/GG'].mean(),
            'Vis/GG': disp['Vis/GG'].mean(),
        }])
        disp = pd.concat([disp, total_row], ignore_index=True)

        # Formattazione valori
        disp_fmt = disp.copy()
        for col in ['Clienti', 'A', 'B', 'C', 'D', 'Volumi A-B-C', 'Volumi D']:
            disp_fmt[col] = disp_fmt[col].apply(lambda x: fmt_eu(x, 0))
        for col in ['Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Min/GG']:
            disp_fmt[col] = disp_fmt[col].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Sat %'] = disp_fmt['Sat %'].apply(lambda x: fmt_eu(x, 1, '%'))
        disp_fmt['Vis/GG'] = disp_fmt['Vis/GG'].apply(lambda x: fmt_eu(x, 2))

        # Genera HTML tabella con stile completo
        def get_sat_color(v):
            try:
                clean = str(v).replace('%', '').replace('.', '').replace(',', '.')
                n = float(clean)
                if n > 110: return '#ffcdd2', '#b71c1c'
                elif n > 100: return '#ffe0b2', '#e65100'
                elif n > 85: return '#fff9c4', '#f57f17'
                else: return '#c8e6c9', '#1b5e20'
            except:
                return '', ''

        # Costruisci HTML manualmente per controllo totale
        html_rows = []
        headers = list(disp_fmt.columns)

        for idx, row in disp_fmt.iterrows():
            is_total = (idx == len(disp_fmt) - 1)
            cells = []
            for col in headers:
                val = row[col]
                if col == 'Sat %':
                    bg, fg = get_sat_color(val)
                    if is_total:
                        style = f'background-color:{bg};color:{fg};font-weight:bold;border-top:2px solid #fff;'
                    else:
                        style = f'background-color:{bg};color:{fg};'
                elif is_total:
                    style = 'background-color:#444444;color:#ffffff;font-weight:bold;border-top:2px solid #fff;'
                else:
                    style = ''
                cells.append(f'<td style="{style}">{val}</td>')

            if is_total:
                html_rows.append(f'<tr style="font-weight:bold;">' + ''.join(cells) + '</tr>')
            else:
                html_rows.append('<tr>' + ''.join(cells) + '</tr>')

        header_cells = []
        for col in headers:
            header_cells.append(f'<th style="background-color:#333333;color:#ffffff;font-weight:bold;text-align:center;padding:8px 4px;border-bottom:2px solid #555;">{col}</th>')

        html_table = (
            '<style>'
            '.kpi-table { border-collapse: collapse; width: 100%; font-family: "Source Sans Pro", sans-serif; font-size: 14px; color: #ffffff; }'
            '.kpi-table th { position: sticky; top: 0; z-index: 1; }'
            '.kpi-table td { color: #ffffff; text-align: center; padding: 6px 4px; }'
            '.kpi-table tr { background-color: transparent; }'
            '</style>'
            '<table class="kpi-table">'
            '<thead><tr>' + ''.join(header_cells) + '</tr></thead>'
            '<tbody>' + ''.join(html_rows) + '</tbody>'
            '</table>'
        )

        st.markdown(html_table, unsafe_allow_html=True)

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
                            opacity=0.25,
                            name=f"Zona {rep}",
                            hoverinfo='name'
                        ))
            classe_colors = {'A': '#ff0000', 'B': '#ffa500', 'C': '#0088ff'}
            for classe, color, size in [('A', classe_colors['A'], 6), ('B', classe_colors['B'], 5), ('C', classe_colors['C'], 4)]:
                df_cl = df_map[df_map['classe'] == classe]
                if len(df_cl) > 0:
                    fig.add_trace(go.Scattermapbox(
                        lat=df_cl['latitudine'], lon=df_cl['longitudine'],
                        mode='markers', marker=dict(size=size, color=color, opacity=0.8),
                        name=f"Classe {classe}",
                        text=df_cl['assigned_rep'].values,
                        hoverinfo='name+text'
                    ))
            df_v_active = df_v_curr[df_v_curr['sales rep'].isin(params['active_list'])].dropna(
                subset=['latitudine', 'longitudine']
            )
            if len(df_v_active) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=df_v_active['latitudine'],
                    lon=df_v_active['longitudine'],
                    mode='markers',
                    marker=dict(size=14, color='black', opacity=0.9),
                    name='🏠 Home Base',
                    hoverinfo='name'
                ))
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=5.5,
                mapbox_center=dict(lat=42.5, lon=12.5),
                margin=dict(r=0, t=30, l=0, b=120),
                height=650,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.subheader("💾 Export Dati")
        export_df = res.copy()
        export_df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        csv = export_df.to_csv(index=False, sep=';', decimal=',')
        st.download_button("📥 Scarica Report CSV", csv, f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)

        # =============================================================================
        # CONFRONTO PRE / POST DOWNSIZE
        # =============================================================================
        if 'pre_downsize_result' in st.session_state and st.session_state.pre_downsize_result is not None:
            st.divider()
            st.subheader("📊 Confronto Prima / Dopo Downsize")
            st.caption("Confronto tra lo scenario immediatamente precedente al downsize e lo scenario attuale.")

            pre_res = st.session_state.pre_downsize_result
            post_res = res  # risultato corrente

            c_pre1, c_pre2, c_pre3, c_pre4 = st.columns(4)
            with c_pre1:
                delta_v = len(post_res) - len(pre_res)
                st.metric("Venditori Attivi", f"{len(pre_res)}", f"{delta_v:+d}")
            with c_pre2:
                delta_sat_m = post_res['saturazione_pct'].mean() - pre_res['saturazione_pct'].mean()
                st.metric("Sat. Media", f"{pre_res['saturazione_pct'].mean():.1f}%", f"{delta_sat_m:+.1f}%")
            with c_pre3:
                delta_sat_x = post_res['saturazione_pct'].max() - pre_res['saturazione_pct'].max()
                st.metric("Sat. Massima", f"{pre_res['saturazione_pct'].max():.1f}%", f"{delta_sat_x:+.1f}%")
            with c_pre4:
                delta_km = post_res['km_annui'].sum() - pre_res['km_annui'].sum()
                st.metric("Km Annui Totali", f"{pre_res['km_annui'].sum():,.0f}", f"{delta_km:+,.0f}")

            # Tabella dettaglio delta per venditore
            merge_compare = pre_res[['sales_rep','saturazione_pct','n_clienti','km_annui']].rename(
                columns={'saturazione_pct':'sat_pre','n_clienti':'cli_pre','km_annui':'km_pre'}
            ).merge(
                post_res[['sales_rep','saturazione_pct','n_clienti','km_annui']].rename(
                    columns={'saturazione_pct':'sat_post','n_clienti':'cli_post','km_annui':'km_post'}
                ), on='sales_rep', how='outer'
            ).fillna(0)

            merge_compare['Δ Sat'] = merge_compare['sat_post'] - merge_compare['sat_pre']
            merge_compare['Δ Clienti'] = merge_compare['cli_post'] - merge_compare['cli_pre']
            merge_compare['Δ Km'] = merge_compare['km_post'] - merge_compare['km_pre']

            st.dataframe(
                merge_compare[['sales_rep','sat_pre','sat_post','Δ Sat','cli_pre','cli_post','Δ Clienti','km_pre','km_post','Δ Km']]
                .rename(columns={
                    'sales_rep':'Venditore','sat_pre':'Sat Pre','sat_post':'Sat Post',
                    'cli_pre':'Cli Pre','cli_post':'Cli Post',
                    'km_pre':'Km Pre','km_post':'Km Post'
                })
                .style.applymap(lambda v: 'color: #e94560; font-weight: bold' if isinstance(v, float) and v > 0 else 'color: #4ecca3' if isinstance(v, float) and v < 0 else '', subset=['Δ Sat','Δ Clienti','Δ Km']),
                use_container_width=True, hide_index=True
            )

            if st.button("🗑️ Chiudi Confronto (libera memoria)", use_container_width=True):
                del st.session_state['pre_downsize_result']
                if 'pre_downsize_df_work' in st.session_state:
                    del st.session_state['pre_downsize_df_work']
                if 'pre_downsize_params' in st.session_state:
                    del st.session_state['pre_downsize_params']
                st.rerun()

    else:
        st.info("📂 Carica Excel e clicca '🚀 Lancia Simulazione' per iniziare.")

if __name__ == "__main__":
    main()
