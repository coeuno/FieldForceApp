import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import warnings
import zipfile
from io import BytesIO
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

def sat_emoji(val):
    if val is None or pd.isna(val):
        return "—"
    try:
        v = float(val)
        if v > 110: return f"{v:.1f}% 🔴"
        elif v > 100: return f"{v:.1f}% 🟠"
        elif v > 85: return f"{v:.1f}% 🟡"
        else: return f"{v:.1f}% 🟢"
    except:
        return str(val)

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
# STILE MAPPA & COLORI LAYOUT
# =============================================================================
# open-street-map  → confini regionali/provinciali ben visibili, colori reali (DEFAULT)
# carto-darkmatter → dark mode, massimo contrasto per i dati sovrapposti
MAPBOX_STYLE = "open-street-map"

def _map_layout_colors():
    """Restituisce colori di layout coerenti con lo stile mappa scelto."""
    if 'dark' in MAPBOX_STYLE:
        return {
            'paper_bg': '#1a1a1a',
            'plot_bg': '#1a1a1a',
            'font_color': '#ffffff',
            'legend_bg': 'rgba(30,30,30,0.85)'
        }
    return {
        'paper_bg': 'white',
        'plot_bg': 'white',
        'font_color': '#333333',
        'legend_bg': 'rgba(255,255,255,0.95)'
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
                   max_stops_per_day):
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

    # Assegnazione venditori (sempre assegnazione attuale dal file Excel o da riassegnazione)
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
# FUNZIONI DOWNSIZE
# =============================================================================
def rank_receivers(orphan_df, df_c, df_v, current_result, active_reps, max_km=9999):
    """
    Rank riceventi per combinazione distanza (dal centroide del loro bacino esistente) / saturazione.
    La distanza media dei clienti orfani è calcolata dal centroide dei clienti che il ricevente
    già visita, non dalla sua casa base. Questo è molto più realistico per un tour di lavoro.
    """
    df_v_valid = df_v.dropna(subset=['latitudine', 'longitudine'])
    scores = []
    for rep in active_reps:
        rep_rows = df_v_valid[df_v_valid['sales rep'] == rep]
        if len(rep_rows) == 0:
            continue

        # --- CENTROIDE DEL BACINO ESISTENTE ---
        rep_clients = df_c[df_c['sales rep'] == rep].dropna(subset=['latitudine', 'longitudine'])
        if len(rep_clients) > 0:
            centroid_lat = rep_clients['latitudine'].mean()
            centroid_lon = rep_clients['longitudine'].mean()
        else:
            rep_row = rep_rows.iloc[0]
            centroid_lat = rep_row['latitudine']
            centroid_lon = rep_row['longitudine']

        # Distanza media dai clienti orfani al centroide del bacino del ricevente
        dists = orphan_df.apply(
            lambda row: haversine_km(row['longitudine'], row['latitudine'], centroid_lon, centroid_lat),
            axis=1
        )
        avg_dist = dists.mean() if len(dists) > 0 else 9999

        # Filtra per max_km
        if avg_dist > max_km:
            continue

        # Saturazione attuale
        if current_result is not None and rep in current_result['sales_rep'].values:
            sat = current_result[current_result['sales_rep'] == rep]['saturazione_pct'].iloc[0]
        else:
            sat = 50.0

        cap_residua = max(0.0, 100.0 - sat)

        # Score: più alto = migliore
        dist_score = max(0.0, 1.0 - avg_dist / 300.0)
        sat_score = cap_residua / 100.0
        score = 0.4 * dist_score + 0.6 * sat_score

        scores.append((rep, score, avg_dist, sat, cap_residua))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores



def preview_allocation(orphan_df, selected_receivers, df_c, df_v):
    """
    Preview: ogni cliente orfano va al ricevente più vicino tra i selezionati.
    La distanza è calcolata dal centroide del bacino esistente di ogni ricevente.
    """
    # Calcola centroide per ogni ricevente
    v_centroids = {}
    for r in selected_receivers:
        rep_clients = df_c[df_c['sales rep'] == r].dropna(subset=['latitudine', 'longitudine'])
        if len(rep_clients) > 0:
            v_centroids[r] = (
                rep_clients['latitudine'].mean(),
                rep_clients['longitudine'].mean()
            )
        else:
            # Fallback: casa base
            rep_rows = df_v.dropna(subset=['latitudine', 'longitudine'])
            rep_rows = rep_rows[rep_rows['sales rep'] == r]
            if len(rep_rows) > 0:
                v_centroids[r] = (rep_rows.iloc[0]['latitudine'], rep_rows.iloc[0]['longitudine'])
            else:
                continue

    allocation = {r: [] for r in selected_receivers if r in v_centroids}
    for idx, row in orphan_df.iterrows():
        best_r = None
        best_d = float('inf')
        for r in selected_receivers:
            if r not in v_centroids:
                continue
            d = haversine_km(
                row['longitudine'], row['latitudine'],
                v_centroids[r][1], v_centroids[r][0]
            )
            if d < best_d:
                best_d = d
                best_r = r
        if best_r:
            allocation[best_r].append((idx, best_d))
    return allocation



def apply_reassignment(df_c, removed_rep, selected_receivers, df_v):
    """Applica la riassegnazione: ogni cliente orfano al ricevente più vicino (dal centroide bacino)."""
    df = df_c.copy()
    orphan_mask = df['sales rep'] == removed_rep
    if not orphan_mask.any():
        return df

    orphan_df = df[orphan_mask].copy()

    # Calcola centroidi bacini riceventi
    v_centroids = {}
    for r in selected_receivers:
        rep_clients = df[df['sales rep'] == r].dropna(subset=['latitudine', 'longitudine'])
        if len(rep_clients) > 0:
            v_centroids[r] = (
                rep_clients['latitudine'].mean(),
                rep_clients['longitudine'].mean()
            )
        else:
            rep_rows = df_v.dropna(subset=['latitudine', 'longitudine'])
            rep_rows = rep_rows[rep_rows['sales rep'] == r]
            if len(rep_rows) > 0:
                v_centroids[r] = (rep_rows.iloc[0]['latitudine'], rep_rows.iloc[0]['longitudine'])

    for idx in orphan_df.index:
        row = df.loc[idx]
        best_r = None
        best_d = float('inf')
        for r in selected_receivers:
            if r not in v_centroids:
                continue
            d = haversine_km(
                row['longitudine'], row['latitudine'],
                v_centroids[r][1], v_centroids[r][0]
            )
            if d < best_d:
                best_d = d
                best_r = r
        if best_r:
            df.loc[idx, 'sales rep'] = best_r

    return df



def render_html_table(headers, rows, font_size="12px"):
    """Renderizza una tabella HTML con bordi e stile."""
    header_html = "".join([
        f'<th style="border:1px solid #555;background:#333;color:#fff;padding:6px 8px;text-align:center;font-size:{font_size};">{h}</th>'
        for h in headers
    ])
    rows_html = ""
    for row in rows:
        cells = "".join([
            f'<td style="border:1px solid #444;padding:5px 8px;text-align:center;font-size:{font_size};">{cell}</td>'
            for cell in row
        ])
        rows_html += f'<tr>{cells}</tr>'

    return (
        f'<table style="border-collapse:collapse;width:100%;margin:8px 0;">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )

# =============================================================================
# NUOVE FUNZIONI: MAPPE & EXPORT
# =============================================================================

# Config Plotly per download PNG nativo dal browser (funziona SEMPRE)
PLOTLY_EXPORT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'mappa',
        'height': 1000,
        'width': 1600,
        'scale': 2
    },
    'displayModeBar': True,
    'displaylogo': False
}

def build_territory_map(df_work, df_v, active_reps, title=""):
    """Genera la mappa territori Plotly (ConvexHull + scatter clienti + home base)."""
    map_colors = _map_layout_colors()
    df_map = df_work.dropna(subset=['latitudine', 'longitudine', 'assigned_rep'])
    fig = go.Figure()
    if len(df_map) == 0:
        return fig

    palette = px.colors.qualitative.Alphabet
    rep_colors = {r: palette[i % len(palette)] for i, r in enumerate(active_reps)}

    for rep in active_reps:
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

    df_v_active = df_v[df_v['sales rep'].isin(active_reps)].dropna(subset=['latitudine', 'longitudine'])
    if len(df_v_active) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=df_v_active['latitudine'],
            lon=df_v_active['longitudine'],
            mode='markers',
            marker=dict(
                size=18,
                symbol='home',
                color='#E63946',
                opacity=1.0,
                line=dict(color='white', width=2.5)
            ),
            text=df_v_active['sales rep'].values,
            name='🏠 Home Base',
            hoverinfo='name+text'
        ))

    fig.update_layout(
        mapbox_style=MAPBOX_STYLE,
        mapbox_zoom=5.5,
        mapbox_center=dict(lat=42.5, lon=12.5),
        margin=dict(r=0, t=30, l=0, b=120),
        height=700,
        title=title,
        paper_bgcolor=map_colors['paper_bg'],
        plot_bgcolor=map_colors['plot_bg'],
        font=dict(color=map_colors['font_color']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor=map_colors['legend_bg'],
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    return fig


def generate_reassignment_map(orphan_df, allocation, df_v, removed_rep, receivers, title=""):
    """Mappa di riassegnazione con frecce cliente → ricevente e home base venditore rimosso."""
    map_colors = _map_layout_colors()
    fig = go.Figure()

    # Home base venditore rimosso (croce rossa)
    removed_home = df_v[df_v['sales rep'] == removed_rep].dropna(subset=['latitudine', 'longitudine'])
    if len(removed_home) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=removed_home['latitudine'].tolist(),
            lon=removed_home['longitudine'].tolist(),
            mode='markers',
            marker=dict(
                size=22,
                symbol='cross',
                color='#D00000',
                opacity=1.0,
                line=dict(color='white', width=2.5)
            ),
            name=f'❌ {removed_rep} (rimosso)'
        ))

    palette = px.colors.qualitative.Bold
    for i, receiver in enumerate(receivers):
        color = palette[i % len(palette)]
        recv_home = df_v[df_v['sales rep'] == receiver].dropna(subset=['latitudine', 'longitudine'])
        if len(recv_home) == 0:
            continue
        recv_lat = recv_home.iloc[0]['latitudine']
        recv_lon = recv_home.iloc[0]['longitudine']

        # Home base ricevente
        fig.add_trace(go.Scattermapbox(
            lat=[recv_lat],
            lon=[recv_lon],
            mode='markers',
            marker=dict(
                size=18,
                symbol='home',
                color=color,
                opacity=1.0,
                line=dict(color='white', width=2.5)
            ),
            name=f'🏠 {receiver}'
        ))

        # Clienti assegnati a questo ricevente
        if receiver in allocation and len(allocation[receiver]) > 0:
            assigned_indices = [idx for idx, _ in allocation[receiver]]
            sub = orphan_df.loc[assigned_indices]

            # Linee (frecce) da cliente a ricevente
            for idx, row in sub.iterrows():
                fig.add_trace(go.Scattermapbox(
                    mode='lines',
                    lat=[row['latitudine'], recv_lat],
                    lon=[row['longitudine'], recv_lon],
                    line=dict(width=1.5, color=color),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Marker clienti orfani
            hover_text = []
            for idx, row in sub.iterrows():
                nome = row.get('ragione sociale', row.get('cliente', f'Cliente {idx}'))
                hover_text.append(f"{nome}<br>→ {receiver}")

            fig.add_trace(go.Scattermapbox(
                lat=sub['latitudine'],
                lon=sub['longitudine'],
                mode='markers',
                marker=dict(size=9, color=color, opacity=0.9),
                name=f'Clienti → {receiver}',
                text=hover_text,
                hoverinfo='text'
            ))

    fig.update_layout(
        mapbox_style=MAPBOX_STYLE,
        mapbox_zoom=5.5,
        mapbox_center=dict(lat=42.5, lon=12.5),
        margin=dict(r=0, t=40, l=0, b=120),
        height=700,
        title=title or f"Riassegnazione: {removed_rep}",
        paper_bgcolor=map_colors['paper_bg'],
        plot_bgcolor=map_colors['plot_bg'],
        font=dict(color=map_colors['font_color']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor=map_colors['legend_bg'],
            bordercolor='gray',
            borderwidth=1
        )
    )
    return fig


def fig_to_png(fig):
    """Converte figura Plotly in bytes PNG. Richiede kaleido."""
    try:
        import plotly.io as pio
        # Test se kaleido è davvero disponibile
        pio.to_image(go.Figure(), format='png')
        img_bytes = pio.to_image(fig, format="png", width=1600, height=1000, scale=2)
        return img_bytes
    except Exception:
        return None


def fig_to_html_bytes(fig):
    """Converte figura Plotly in bytes HTML interattivo (fallback senza kaleido)."""
    buffer = BytesIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    buffer.seek(0)
    return buffer.getvalue()


def generate_excel_report(initial_result, final_result, reassignment_history, removal_details, col_vol):
    """Genera Excel multi-foglio: Iniziale, Finale, Confronto, Dettaglio, Variazioni."""
    if initial_result is None or final_result is None:
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Stato Iniziale
        init_sheet = initial_result.copy()
        init_sheet.to_excel(writer, sheet_name='Stato Iniziale', index=False)

        # 2. Stato Finale
        final_sheet = final_result.copy()
        final_sheet.to_excel(writer, sheet_name='Stato Finale', index=False)

        # 3. Confronto Orizzontale
        # Colonne = venditori (prima tutti quelli iniziali, poi tutti quelli finali)
        all_reps = sorted(set(init_sheet['sales_rep'].tolist() + final_sheet['sales_rep'].tolist()))
        metrics = ['n_clienti', 'n_classe_a', 'n_classe_b', 'n_classe_c', 'n_classe_d',
                   'volume_abc', 'volume_d', 'ore_visite_annue', 'ore_viaggio_annue',
                   'ore_totali_annue', 'saturazione_pct', 'km_annui', 'driving_min_giorno', 'visite_giorno']
        metric_labels = ['N Clienti', 'N Classe A', 'N Classe B', 'N Classe C', 'N Classe D',
                         'Volume ABC', 'Volume D', 'Ore Visite', 'Ore Viaggio',
                         'Ore Totali', 'Saturazione %', 'KM Annui', 'Min/GG', 'Vis/GG']

        confronto_data = {}
        for rep in all_reps:
            init_row = init_sheet[init_sheet['sales_rep'] == rep]
            init_vals = init_row.iloc[0] if len(init_row) > 0 else {m: 0 for m in metrics}
            confronto_data[f"{rep}_INIZIALE"] = [init_vals.get(m, 0) for m in metrics]

        for rep in all_reps:
            final_row = final_sheet[final_sheet['sales_rep'] == rep]
            final_vals = final_row.iloc[0] if len(final_row) > 0 else {m: 0 for m in metrics}
            confronto_data[f"{rep}_FINALE"] = [final_vals.get(m, 0) for m in metrics]

        confronto_df = pd.DataFrame(confronto_data, index=metric_labels)
        confronto_df.to_excel(writer, sheet_name='Confronto Orizzontale')

        # 4. Dettaglio Riassegnazioni (con provenienza)
        dettagli_rows = []
        for hist in reassignment_history:
            removed = hist['removed']
            if removed in removal_details:
                details = removal_details[removed]
                alloc = details['allocation']
                orphan_df = details['orphan_df']
                for receiver, client_list in alloc.items():
                    for client_idx, dist in client_list:
                        if client_idx in orphan_df.index:
                            cliente = orphan_df.loc[client_idx]
                            dettagli_rows.append({
                                'Step_Rimozione': removed,
                                'Ricevente': receiver,
                                'Cliente_Index': client_idx,
                                'Ragione_Sociale': cliente.get('ragione sociale', 'N/D'),
                                'Sigla': cliente.get('sigla', 'N/D'),
                                'Classe': cliente.get('classe', 'N/D'),
                                'Volume': cliente.get(col_vol, 0),
                                'Distanza_Ricevente_km': round(dist, 2),
                                'Lat': cliente.get('latitudine', ''),
                                'Lon': cliente.get('longitudine', '')
                            })
        if dettagli_rows:
            pd.DataFrame(dettagli_rows).to_excel(writer, sheet_name='Dettaglio Riassegnazioni', index=False)

        # 5. Variazioni
        variazioni = []
        for rep in all_reps:
            init_row = init_sheet[init_sheet['sales_rep'] == rep]
            final_row = final_sheet[final_sheet['sales_rep'] == rep]
            init_vals = init_row.iloc[0] if len(init_row) > 0 else None
            final_vals = final_row.iloc[0] if len(final_row) > 0 else None

            row = {'Venditore': rep}
            for m, label in zip(metrics, metric_labels):
                v_init = init_vals[m] if init_vals is not None else 0
                v_final = final_vals[m] if final_vals is not None else 0
                row[f"{label}_INI"] = v_init
                row[f"{label}_FIN"] = v_final
                if abs(v_init) > 0.001:
                    row[f"{label}_DELTA_%"] = round(((v_final - v_init) / v_init) * 100, 2)
                else:
                    row[f"{label}_DELTA_%"] = 0.0 if abs(v_final) < 0.001 else 999.0
            variazioni.append(row)
        pd.DataFrame(variazioni).to_excel(writer, sheet_name='Variazioni', index=False)

        # 6. Riepilogo Step
        if reassignment_history:
            pd.DataFrame(reassignment_history).to_excel(writer, sheet_name='Riepilogo Step', index=False)

    output.seek(0)
    return output

# =============================================================================
# INTERFACCIA
# =============================================================================
def main():
    # Inizializzazione stato sessione
    if 'df_c_original' not in st.session_state: st.session_state.df_c_original = None
    if 'df_c_working' not in st.session_state: st.session_state.df_c_working = None
    if 'removed_reps' not in st.session_state: st.session_state.removed_reps = []
    if 'reassignment_history' not in st.session_state: st.session_state.reassignment_history = []
    if 'pending_removal' not in st.session_state: st.session_state.pending_removal = None
    if 'rep_status_prev' not in st.session_state: st.session_state.rep_status_prev = {}
    if 'force_recalc' not in st.session_state: st.session_state.force_recalc = False
    if 'current_result' not in st.session_state: st.session_state.current_result = None
    if 'current_df_work' not in st.session_state: st.session_state.current_df_work = None
    if 'current_params' not in st.session_state: st.session_state.current_params = {}
    if 'current_df_v' not in st.session_state: st.session_state.current_df_v = None
    if 'abc_version' not in st.session_state: st.session_state.abc_version = 2
    if 'abc_vals' not in st.session_state: st.session_state.abc_vals = None
    if 'min_vol' not in st.session_state: st.session_state.min_vol = 0
    if 'max_km_filter' not in st.session_state: st.session_state.max_km_filter = 200

    # --- NUOVO: stato per export avanzato ---
    if 'initial_result' not in st.session_state: st.session_state.initial_result = None
    if 'initial_df_work' not in st.session_state: st.session_state.initial_df_work = None
    if 'initial_fig' not in st.session_state: st.session_state.initial_fig = None
    if 'removal_figures' not in st.session_state: st.session_state.removal_figures = {}
    if 'removal_details' not in st.session_state: st.session_state.removal_details = {}
    if 'final_fig' not in st.session_state: st.session_state.final_fig = None

    # --- RILEVAMENTO KALEIDO ---
    KALEIDO_AVAILABLE = False
    try:
        import plotly.io as pio
        pio.to_image(go.Figure(), format='png')
        KALEIDO_AVAILABLE = True
    except Exception:
        KALEIDO_AVAILABLE = False
    st.session_state['kaleido_available'] = KALEIDO_AVAILABLE

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
            st.session_state.df_c_original = None
            st.session_state.df_c_working = None
            st.session_state.removed_reps = []
            st.session_state.reassignment_history = []
            st.session_state.pending_removal = None
            st.session_state.rep_status_prev = {}
            st.session_state.trigger_auto_run = True
            st.session_state.current_result = None
            st.session_state.current_df_work = None
            st.session_state.current_params = {}
            st.session_state.current_df_v = None
            # --- reset export state ---
            st.session_state.initial_result = None
            st.session_state.initial_df_work = None
            st.session_state.initial_fig = None
            st.session_state.removal_figures = {}
            st.session_state.removal_details = {}
            st.session_state.final_fig = None
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

        # Inizializza working set se necessario
        if st.session_state.df_c_original is None:
            st.session_state.df_c_original = df_c.copy()
        if st.session_state.df_c_working is None:
            st.session_state.df_c_working = df_c.copy()

        st.success(f"✅ {len(df_c):,} clienti, {len(df_v):,} venditori caricati")
        clienti_con_rep = st.session_state.df_c_working['sales rep'].notna().sum()
        st.caption(f"📍 {clienti_con_rep:,} clienti hanno un sales rep assegnato")
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
        removed_reps = st.session_state.get('removed_reps', [])
        with st.expander("Attiva / Disattiva venditori", expanded=True):
            rep_status = {}
            for r in reps:
                if r in removed_reps:
                    st.checkbox(f"~~{r}~~ (rimosso)", value=False, disabled=True, key=f"rep_{r}")
                    rep_status[r] = False
                else:
                    default_val = st.session_state.rep_status_prev.get(r, True)
                    rep_status[r] = st.checkbox(r, value=default_val, key=f"rep_{r}")

    # =============================================================================
    # DOPO SIDEBAR: usa il working set come fonte di verità
    # =============================================================================
    df_c = st.session_state.df_c_working

    # =============================================================================
    # RILEVAMENTO RIMOZIONE VENDITORE
    # =============================================================================
    if uploaded and not st.session_state.get('pending_removal'):
        for r in reps:
            if r not in removed_reps:
                prev_val = st.session_state.rep_status_prev.get(r, True)
                curr_val = rep_status[r]
                if prev_val and not curr_val:
                    orphan = st.session_state.df_c_working[
                        st.session_state.df_c_working['sales rep'] == r
                    ].copy()
                    st.session_state.pending_removal = {
                        'rep': r,
                        'orphan_clients': orphan
                    }
                    st.rerun()

    # =============================================================================
    # PANNELLO DI RIASSEGNAZIONE
    # =============================================================================
    if st.session_state.get('pending_removal'):
        rem = st.session_state.pending_removal
        removed_name = rem['rep']
        orphan_df = rem['orphan_clients']
        n_orfani = len(orphan_df)

        st.divider()
        st.subheader(f"🔴 Riassegnazione richiesta: {removed_name}")

        if n_orfani == 0:
            st.warning(f"⚠️ {removed_name} non ha clienti assegnati. Rimozione immediata.")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("✅ Conferma Rimozione", use_container_width=True):
                    st.session_state.removed_reps.append(removed_name)
                    st.session_state.reassignment_history.append({
                        'removed': removed_name,
                        'receivers': [],
                        'n_clients': 0,
                        'timestamp': pd.Timestamp.now().strftime("%H:%M:%S")
                    })
                    st.session_state.pending_removal = None
                    st.session_state.force_recalc = True
                    st.session_state.rep_status_prev = {r: rep_status[r] for r in reps}
                    st.rerun()
            with c2:
                if st.button("❌ Annulla", use_container_width=True):
                    st.session_state[f"rep_{removed_name}"] = True
                    st.session_state.pending_removal = None
                    st.rerun()
            st.stop()

        # --- TABELLA CLIENTI ORFANI (ORIZZONTALE) ---
        orphan_df = classify_abc(orphan_df, col_vol, 
                                  st.session_state.abc_vals['da_a'],
                                  st.session_state.abc_vals['da_b'],
                                  st.session_state.abc_vals['da_c'])

        n_a = int((orphan_df['classe'] == 'A').sum())
        n_b = int((orphan_df['classe'] == 'B').sum())
        n_c = int((orphan_df['classe'] == 'C').sum())
        n_d = int((orphan_df['classe'] == 'D').sum())

        vol_a = float(orphan_df.loc[orphan_df['classe'] == 'A', col_vol].sum()) if n_a > 0 else 0.0
        vol_b = float(orphan_df.loc[orphan_df['classe'] == 'B', col_vol].sum()) if n_b > 0 else 0.0
        vol_c = float(orphan_df.loc[orphan_df['classe'] == 'C', col_vol].sum()) if n_c > 0 else 0.0
        vol_d = float(orphan_df.loc[orphan_df['classe'] == 'D', col_vol].sum()) if n_d > 0 else 0.0

        st.markdown(f"### 📍 {n_orfani} clienti orfani da {removed_name}")

        # Tabella orizzontale: colonne = classi con emoji colorati, righe = metriche
        orphan_headers = ['Metrica', '🟢 Cliente A', '🟡 Cliente B', '🔴 Cliente C', '⚫ Cliente D', '🔵 Totali']
        orphan_rows = [
            ['N. clienti', str(n_a), str(n_b), str(n_c), str(n_d), f'**{n_orfani}**'],
            ['Volume', fmt_eu(vol_a), fmt_eu(vol_b), fmt_eu(vol_c), fmt_eu(vol_d), 
             f'**{fmt_eu(vol_a + vol_b + vol_c + vol_d)}**']
        ]
        st.markdown(render_html_table(orphan_headers, orphan_rows, "13px"), unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🎯 Seleziona i riceventi")

        active_reps = [r for r in reps if r != removed_name and r not in removed_reps]

        # Filtro distanza massima
        max_km = st.number_input(
            "Max distanza media (km) per considerare un ricevente",
            min_value=10, max_value=1000,
            value=st.session_state.max_km_filter,
            step=10, key="max_km_input"
        )
        st.session_state.max_km_filter = max_km

        ranked = rank_receivers(orphan_df, st.session_state.df_c_working, df_v, 
                                st.session_state.get('current_result'), active_reps, max_km)

        if not ranked:
            st.warning(f"Nessun ricevente entro {max_km} km. Prova ad aumentare la distanza massima.")
            st.stop()

        # CSS per checkbox rosse (stile sidebar)
        st.markdown("""
        <style>
        [data-testid="stCheckbox"] > label > div[role="checkbox"] {
            background-color: #ff4444 !important;
            border-color: #ff4444 !important;
        }
        [data-testid="stCheckbox"] > label > div[role="checkbox"][aria-checked="true"] {
            background-color: #ff4444 !important;
            border-color: #ff4444 !important;
        }
        [data-testid="stCheckbox"] > label > div[role="checkbox"] > div {
            background-color: #ff4444 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Prepara dati riceventi base (senza saturazione futura precalcolata)
        recv_data = []
        for i, (rep, score, avg_dist, sat, cap) in enumerate(ranked):
            is_suggested = i < 3

            # Clienti A-B-C attuali del ricevente
            rep_current_clients = st.session_state.df_c_working[
                st.session_state.df_c_working['sales rep'] == rep
            ].copy()
            rep_current_clients = classify_abc(
                rep_current_clients, col_vol,
                st.session_state.abc_vals['da_a'],
                st.session_state.abc_vals['da_b'],
                st.session_state.abc_vals['da_c']
            )
            current_abc = int((rep_current_clients['classe'].isin(['A','B','C'])).sum())

            recv_data.append({
                'Seleziona': is_suggested,
                'Ricevente': rep,
                'Clienti A-B-C attuali': current_abc,
                'Sat. Attuale': sat_emoji(sat),
                'Distanza media (km) nuovi clienti': f"{avg_dist:.1f}",
                'Clienti A-B-C aggiuntivi': '—',
                'Sat. Futura': '—'
            })

        editor_key = f"recv_editor_{removed_name}"
        edited = st.data_editor(
            pd.DataFrame(recv_data),
            column_config={
                "Seleziona": st.column_config.CheckboxColumn(
                    "Seleziona",
                    help="Spunta per selezionare questo ricevente",
                    default=False,
                ),
                "Ricevente": st.column_config.TextColumn("Ricevente", disabled=True),
                "Clienti A-B-C attuali": st.column_config.NumberColumn("Clienti A-B-C attuali", disabled=True),
                "Sat. Attuale": st.column_config.TextColumn("Sat. Attuale", disabled=True),
                "Distanza media (km) nuovi clienti": st.column_config.TextColumn(
                    "Distanza media (km) nuovi clienti", disabled=True
                ),
                "Clienti A-B-C aggiuntivi": st.column_config.TextColumn("Clienti A-B-C aggiuntivi", disabled=True),
                "Sat. Futura": st.column_config.TextColumn("Sat. Futura", disabled=True),
            },
            disabled=["Ricevente", "Clienti A-B-C attuali", "Sat. Attuale", 
                      "Distanza media (km) nuovi clienti", "Clienti A-B-C aggiuntivi", "Sat. Futura"],
            hide_index=True,
            use_container_width=True,
            key=editor_key
        )

        selected_receivers = edited[edited['Seleziona']]['Ricevente'].tolist()

        # --- BOTTONE CALCOLO SATURAZIONE FUTURA ---
        calc_col1, calc_col2 = st.columns([1, 3])
        with calc_col1:
            calc_pressed = st.button(
                "🔄 Calcola Impatto", 
                use_container_width=True, 
                disabled=(not selected_receivers),
                key=f"btn_calc_{removed_name}"
            )

        if calc_pressed and selected_receivers:
            with st.spinner("Calcolo scenario in corso... (può richiedere qualche secondo)"):
                abc = st.session_state.abc_vals

                # 1. Applica riassegnazione temporanea con SOLO i riceventi selezionati
                df_temp = apply_reassignment(
                    st.session_state.df_c_working, removed_name, selected_receivers, df_v
                )

                # 2. Tutti i venditori attivi dalla sidebar (inclusi i riceventi selezionati)
                temp_active = [r for r in reps if r not in st.session_state.removed_reps and rep_status.get(r, True)]

                # 3. Simulazione completa UNA SOLA VOLTA
                temp_res, _ = run_simulation(
                    df_temp, df_v, temp_active, col_vol,
                    abc['da_a'], abc['da_b'], abc['da_c'],
                    abc['freq_a'], abc['freq_b'], abc['freq_c'],
                    dur_visita, ore_effettive_gg, gg_lavoro,
                    max_stops_per_day
                )

                # 4. Calcola allocazione per contare clienti aggiuntivi per ogni ricevente
                alloc = preview_allocation(orphan_df, selected_receivers, st.session_state.df_c_working, df_v)

                # 5. Estrai risultati per ogni ricevente selezionato
                future_results = {}
                for rep in selected_receivers:
                    fut_sat = None
                    if temp_res is not None and rep in temp_res['sales_rep'].values:
                        fut_sat = float(temp_res[temp_res['sales_rep'] == rep]['saturazione_pct'].iloc[0])

                    add_abc = 0
                    if rep in alloc and len(alloc[rep]) > 0:
                        assigned_indices = [i for i, _ in alloc[rep]]
                        sub_assigned = orphan_df.loc[assigned_indices]
                        sub_assigned = classify_abc(
                            sub_assigned, col_vol,
                            abc['da_a'], abc['da_b'], abc['da_c']
                        )
                        add_abc = int((sub_assigned['classe'].isin(['A','B','C'])).sum())

                    future_results[rep] = {
                        'fut_sat': fut_sat,
                        'add_abc': add_abc
                    }

                # Salva in session state per visualizzazione
                st.session_state[f"future_results_{removed_name}"] = future_results
                st.session_state[f"last_selected_{removed_name}"] = sorted(selected_receivers)
            st.rerun()

        # --- MOSTRA RISULTATI CALCOLATI ---
        future_results = st.session_state.get(f"future_results_{removed_name}", {})
        last_selected = st.session_state.get(f"last_selected_{removed_name}", [])

        if future_results and sorted(selected_receivers) == last_selected:
            result_rows = []
            for _, row in edited.iterrows():
                rep = row['Ricevente']
                if row['Seleziona'] and rep in future_results:
                    row_dict = row.to_dict()
                    row_dict['Clienti A-B-C aggiuntivi'] = future_results[rep]['add_abc']
                    row_dict['Sat. Futura'] = sat_emoji(future_results[rep]['fut_sat'])
                    result_rows.append(row_dict)
                else:
                    result_rows.append(row.to_dict())

            if result_rows:
                st.markdown("#### 📊 Impatto Simulato (basato sui riceventi selezionati)")
                result_df = pd.DataFrame(result_rows)
                result_df = result_df[['Seleziona', 'Ricevente', 'Clienti A-B-C attuali', 
                                        'Sat. Attuale', 'Distanza media (km) nuovi clienti',
                                        'Clienti A-B-C aggiuntivi', 'Sat. Futura']]
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.caption("La saturazione futura è calcolata riassegnando gli orfani ai soli venditori selezionati e simulando il carico completo.")
        elif future_results and sorted(selected_receivers) != last_selected:
            st.warning("⚠️ La selezione è cambiata rispetto all'ultimo calcolo. Clicca '🔄 Calcola Impatto' per aggiornare.")

        st.divider()
        st.markdown("### 📊 Preview allocazione clienti")

        if not selected_receivers:
            st.warning("Seleziona almeno un ricevente per vedere la preview")
        else:
            alloc = preview_allocation(orphan_df, selected_receivers, st.session_state.df_c_working, df_v)

            preview_headers = ['Ricevente', 'Clienti A', 'Clienti B', 'Clienti C', 'Clienti D', 'Totale Clienti', 'Volume Totale']
            preview_rows = []

            for r in selected_receivers:
                n_ass = len(alloc.get(r, []))
                if n_ass > 0:
                    assigned_indices = [idx for idx, _ in alloc[r]]
                    sub_assigned = orphan_df.loc[assigned_indices]

                    na = int((sub_assigned['classe'] == 'A').sum())
                    nb = int((sub_assigned['classe'] == 'B').sum())
                    nc = int((sub_assigned['classe'] == 'C').sum())
                    nd = int((sub_assigned['classe'] == 'D').sum())
                    vol_tot = sub_assigned[col_vol].sum()

                    preview_rows.append([
                        f"**{r}**",
                        str(na), str(nb), str(nc), str(nd),
                        str(n_ass),
                        fmt_eu(vol_tot)
                    ])

            if preview_rows:
                st.markdown(render_html_table(preview_headers, preview_rows, "12px"), unsafe_allow_html=True)
            else:
                st.info("Nessun cliente assegnato ai riceventi selezionati")

        # --- BOTTONI CONFERMA / ANNULLA ---
        st.divider()
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("✅ Conferma Riassegnazione", use_container_width=True, disabled=(not selected_receivers)):
                if selected_receivers:
                    # --- NUOVO: cattura mappa e dettagli PRIMA di modificare lo stato ---
                    alloc = preview_allocation(orphan_df, selected_receivers, st.session_state.df_c_working, df_v)
                    fig_reassign = generate_reassignment_map(
                        orphan_df, alloc, df_v, removed_name, selected_receivers,
                        title=f"🔴 Step {len(st.session_state.removed_reps)+1}: {removed_name} → {', '.join(selected_receivers)}"
                    )
                    st.session_state.removal_figures[removed_name] = fig_reassign
                    st.session_state.removal_details[removed_name] = {
                        'allocation': alloc,
                        'orphan_df': orphan_df.copy(),
                        'receivers': selected_receivers,
                        'timestamp': pd.Timestamp.now().strftime("%H:%M:%S")
                    }
                    # --- FINE NUOVO ---

                    df_new = apply_reassignment(
                        st.session_state.df_c_working, removed_name, selected_receivers, df_v
                    )
                    st.session_state.df_c_working = df_new
                    st.session_state.removed_reps.append(removed_name)
                    st.session_state.reassignment_history.append({
                        'removed': removed_name,
                        'receivers': selected_receivers,
                        'n_clients': n_orfani,
                        'timestamp': pd.Timestamp.now().strftime("%H:%M:%S")
                    })
                    # Pulisci cache riassegnazione
                    st.session_state.pop(f"future_results_{removed_name}", None)
                    st.session_state.pop(f"last_selected_{removed_name}", None)
                    st.session_state.pending_removal = None
                    st.session_state.force_recalc = True
                    st.session_state.rep_status_prev = {r: rep_status[r] for r in reps}
                    st.rerun()
        with col_btn2:
            if st.button("❌ Annulla Rimozione", use_container_width=True):
                st.session_state[f"rep_{removed_name}"] = True
                st.session_state.pending_removal = None
                st.rerun()

        st.stop()

    # =============================================================================
    # MAIN CONTENT: Matrice ABC
    # =============================================================================
    if uploaded:
        st.divider()
        st.subheader("📊 Matrice Classificazione ABC & Frequenze")
        st.caption("Definisci gli intervalli esatti (da/a) e le visite annue per ogni classe.")

        if st.session_state.abc_vals is None:
            st.session_state.abc_vals = {
                'da_a': 801, 'a_a': -1, 'freq_a': 24,
                'da_b': 401, 'a_b': 800, 'freq_b': 16,
                'da_c': 101, 'a_c': 400, 'freq_c': 12,
                'da_d': 0, 'a_d': 100, 'freq_d': 0
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
        st.session_state.min_vol = da_c

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
    run_sim = False
    if st.session_state.get('trigger_auto_run', False):
        st.session_state.trigger_auto_run = False
        run_sim = True
    if st.session_state.get('force_recalc', False):
        st.session_state.force_recalc = False
        run_sim = True
    if manual_run: 
        run_sim = True

    if run_sim and uploaded:
        with st.spinner("🔄 Calcolo scenario in corso..."):
            try:
                active_list = [r for r, s in rep_status.items() if s]
                if len(active_list) == 0:
                    st.error("⚠️ Seleziona almeno un venditore")
                else:
                    abc = st.session_state.abc_vals
                    res, df_w = run_simulation(df_c, df_v, active_list, col_vol, 
                                               abc['da_a'], abc['da_b'], abc['da_c'],
                                               abc['freq_a'], abc['freq_b'], abc['freq_c'],
                                               dur_visita, ore_effettive_gg, gg_lavoro,
                                               max_stops_per_day)
                    if res is None: 
                        st.error(df_w)
                    else:
                        st.session_state.current_result = res
                        st.session_state.current_df_work = df_w
                        st.session_state.current_params = {
                            'active_list': active_list, 'ore_gg': ore_effettive_gg,
                            'gg_lavoro': gg_lavoro, 'dur_visita': dur_visita,
                            'max_stops': max_stops_per_day, 'reps': reps,
                            'min_vol': st.session_state.get('min_vol', 0),
                            'col_vol': col_vol
                        }
                        st.session_state.current_df_v = df_v
                        st.success("✅ Calcolo completato!")

                        # --- NUOVO: cattura stato iniziale se è la prima volta ---
                        if st.session_state.initial_result is None:
                            st.session_state.initial_result = res.copy()
                            st.session_state.initial_df_work = df_w.copy()
                            st.session_state.initial_params = st.session_state.current_params.copy()
                            st.session_state.initial_fig = build_territory_map(
                                df_w, df_v, active_list, title="🗺️ Status Quo Iniziale"
                            )
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

        min_vol_display = st.session_state.get('min_vol', 0)
        st.info(f"📍 Stop/Giorno: **{params['max_stops']}** | Soglia minima: **≥{fmt_eu(min_vol_display)}**")

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
            map_colors = _map_layout_colors()
            palette = px.colors.qualitative.Alphabet
            rep_colors = {r: palette[i % len(palette)] for i, r in enumerate(params['active_list'])}
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
                    marker=dict(
                        size=18,
                        symbol='home',
                        color='#E63946',
                        opacity=1.0,
                        line=dict(color='white', width=2.5)
                    ),
                    text=df_v_active['sales rep'].values,
                    name='🏠 Home Base',
                    hoverinfo='name+text'
                ))
            fig.update_layout(
                mapbox_style=MAPBOX_STYLE,
                mapbox_zoom=5.5,
                mapbox_center=dict(lat=42.5, lon=12.5),
                margin=dict(r=0, t=30, l=0, b=120),
                height=700,
                paper_bgcolor=map_colors['paper_bg'],
                plot_bgcolor=map_colors['plot_bg'],
                font=dict(color=map_colors['font_color']),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    bgcolor=map_colors['legend_bg'],
                    bordercolor='gray',
                    borderwidth=1,
                    font=dict(size=11)
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_EXPORT_CONFIG)

        # =============================================================================
        # NUOVO: EXPORT AVANZATO & REPORTISTICA
        # =============================================================================
        st.divider()
        st.subheader("📦 Export Avanzato & Reportistica")
        
        kaleido_ok = st.session_state.get('kaleido_available', False)
        if not kaleido_ok:
            st.info("ℹ️ **Kaleido non rilevato**: l'export PNG automatico è disabilitato. "
                    "Puoi comunque scaricare PNG cliccando il pulsante 📷 nella barra degli strumenti di ogni mappa. "
                    "I bottoni qui sotto genereranno file HTML interattivi come fallback.")

        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            if st.session_state.initial_fig is not None:
                if kaleido_ok:
                    if st.button("📸 Scarica Mappa Iniziale PNG", use_container_width=True):
                        img = fig_to_png(st.session_state.initial_fig)
                        if img:
                            st.download_button("⬇️ Download PNG", img, "mappa_status_quo_iniziale.png", "image/png", use_container_width=True)
                else:
                    if st.button("📄 Scarica Mappa Iniziale HTML", use_container_width=True):
                        html_bytes = fig_to_html_bytes(st.session_state.initial_fig)
                        st.download_button("⬇️ Download HTML", html_bytes, "mappa_status_quo_iniziale.html", "text/html", use_container_width=True)

        with col_ex2:
            if st.session_state.removal_figures:
                if kaleido_ok:
                    if st.button("📸 Scarica Mappe Riassegnazioni PNG", use_container_width=True):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for rep_name, fig in st.session_state.removal_figures.items():
                                img = fig_to_png(fig)
                                if img:
                                    zf.writestr(f"riassegnazione_{rep_name}.png", img)
                        zip_buffer.seek(0)
                        st.download_button("⬇️ Download ZIP Mappe", zip_buffer.getvalue(), "mappe_riassegnazioni.zip", "application/zip", use_container_width=True)
                else:
                    if st.button("📄 Scarica Mappe Riassegnazioni HTML", use_container_width=True):
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for rep_name, fig in st.session_state.removal_figures.items():
                                html_bytes = fig_to_html_bytes(fig)
                                zf.writestr(f"riassegnazione_{rep_name}.html", html_bytes)
                        zip_buffer.seek(0)
                        st.download_button("⬇️ Download ZIP HTML", zip_buffer.getvalue(), "mappe_riassegnazioni_html.zip", "application/zip", use_container_width=True)

        with col_ex3:
            if st.button("🏁 Status Quo Finale + Export Excel", use_container_width=True, type="primary"):
                with st.spinner("Generazione report finale..."):
                    # Genera mappa finale
                    final_fig = build_territory_map(
                        st.session_state.current_df_work,
                        st.session_state.current_df_v,
                        st.session_state.current_params['active_list'],
                        title="🗺️ Status Quo Finale"
                    )
                    st.session_state.final_fig = final_fig

                    # Genera Excel
                    excel_buffer = generate_excel_report(
                        st.session_state.initial_result,
                        st.session_state.current_result,
                        st.session_state.reassignment_history,
                        st.session_state.removal_details,
                        st.session_state.current_params.get('col_vol', col_vol)
                    )

                    if excel_buffer:
                        st.success("✅ Report generato!")
                        st.download_button("📥 Scarica Excel Completo", excel_buffer,
                                        f"downsizing_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True)
                        if kaleido_ok:
                            img_final = fig_to_png(final_fig)
                            if img_final:
                                st.download_button("📥 Scarica Mappa Finale PNG", img_final, "mappa_status_quo_finale.png", "image/png", use_container_width=True)
                        else:
                            html_final = fig_to_html_bytes(final_fig)
                            st.download_button("📥 Scarica Mappa Finale HTML", html_final, "mappa_status_quo_finale.html", "text/html", use_container_width=True)
                    else:
                        st.error("❌ Errore generazione Excel")

        st.divider()
        st.subheader("💾 Export Dati Base")
        export_df = res.copy()
        export_df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        csv = export_df.to_csv(index=False, sep=';', decimal=',')
        st.download_button("📥 Scarica Report CSV", csv, f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
    else:
        st.info("📂 Carica Excel e clicca '🚀 Lancia Simulazione' per iniziare.")

    # =============================================================================
    # SALVATAGGIO STATO CHECKBOX PER RILEVAMENTO FUTURO
    # =============================================================================
    if uploaded and not st.session_state.get('pending_removal'):
        st.session_state.rep_status_prev = {r: rep_status[r] for r in reps}

if __name__ == "__main__":
    main()
