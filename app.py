import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import warnings
import zipfile
from io import BytesIO
from io import StringIO
import colorsys
from collections import defaultdict
warnings.filterwarnings('ignore')
st.set_page_config(page_title="🎯 Field Force Downsizing Simulator", layout="wide", page_icon="🚀")

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
    if s > 110: return "🔴 CRITIC"
    elif s > 100: return "🟠 OVERLOAD"
    elif s > 85: return "⚠️ WARNING"
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


CONGESTION_FACTOR = {
    "MI": 1.60, "RM": 1.60, "NA": 1.60, "TO": 1.60, "GE": 1.60,
    "BO": 1.60, "FI": 1.60, "VE": 1.60, "BA": 1.60, "CT": 1.60, "PA": 1.60,
    "BG": 1.30, "BS": 1.30, "CO": 1.30, "VA": 1.30, "VR": 1.30, "VI": 1.30,
    "PD": 1.30, "TV": 1.30, "RO": 1.30, "PC": 1.30, "PR": 1.30, "RE": 1.30,
    "MO": 1.30, "FE": 1.30, "RA": 1.30, "RN": 1.30, "FO": 1.30, "FC": 1.30,
    "PU": 1.30, "AN": 1.30, "MC": 1.30, "AP": 1.30, "FM": 1.30, "PE": 1.30,
    "CH": 1.30, "TE": 1.30, "LT": 1.30, "FR": 1.30, "VT": 1.30, "RI": 1.30,
    "LU": 1.30, "PI": 1.30, "LI": 1.30, "PO": 1.30, "PT": 1.30, "AR": 1.30,
    "SI": 1.30, "PG": 1.30, "TR": 1.30, "MS": 1.30, "CA": 1.30, "SS": 1.30,
    "SV": 1.30, "IM": 1.30, "SP": 1.30, "FG": 1.30, "BT": 1.30, "BR": 1.30,
    "TA": 1.30, "LE": 1.30, "KR": 1.30, "RG": 1.30, "SR": 1.30, "ME": 1.30,
    "TP": 1.30, "AG": 1.30, "CL": 1.30, "EN": 1.30, "CS": 1.30, "CZ": 1.30,
    "VV": 1.30, "RC": 1.30, "UD": 1.30, "GO": 1.30, "TS": 1.30,
    "PN": 1.30, "BL": 1.30, "BZ": 1.30, "TN": 1.30, "AO": 1.30, "SO": 1.30,
    "VB": 1.30, "VC": 1.30, "NO": 1.30, "BI": 1.30, "AL": 1.30, "AT": 1.30,
    "CN": 1.30, "CB": 1.30, "IS": 1.30, "CE": 1.30, "BN": 1.30, "AV": 1.30,
    "SA": 1.30, "PZ": 1.30, "MT": 1.30, "AQ": 1.30, "NU": 1.30, "OR": 1.30,
    "OT": 1.30, "SU": 1.30
}

# =============================================================================
# === V2 MODIFIED: Parametri modello a 2 Velocita ===
# =============================================================================
HIGHWAY_CIRCUITY = 1.05
URBAN_RADIUS_KM = 30

HIGHWAY_SPEED_BY_PROVINCE = defaultdict(lambda: 130)
HIGHWAY_SPEED_BY_PROVINCE.update({
    'AO': 100, 'VA': 140, 'BI': 140, 'NO': 140, 'VC': 140, 'TO': 140,
    'GE': 130, 'SV': 130, 'IM': 130, 'SP': 130,
    'BO': 130, 'MO': 130, 'RE': 130, 'PR': 130, 'PC': 130, 'RA': 130, 'FE': 130, 'RN': 130, 'FO': 130, 'FC': 130,
    'FI': 130, 'LI': 130, 'PO': 130, 'PT': 130, 'AR': 130, 'SI': 130, 'GR': 130, 'PG': 130, 'TR': 130,
    'RM': 120, 'LT': 120, 'FR': 120, 'VT': 120, 'RI': 120,
    'NA': 120, 'CE': 120, 'BN': 120, 'AV': 120, 'SA': 120,
    'BA': 130, 'BR': 130, 'TA': 130, 'LE': 130, 'FG': 130, 'BT': 130, 'KR': 130,
    'CT': 120, 'PA': 120, 'ME': 120, 'AG': 120, 'CL': 120, 'EN': 120, 'RG': 120, 'SR': 120, 'TP': 120,
    'CA': 120, 'SS': 120, 'NU': 120, 'OR': 120, 'OT': 120, 'SU': 120,
    'VE': 130, 'PD': 130, 'RO': 130, 'VR': 130, 'VI': 130, 'TV': 130, 'BL': 130,
    'TN': 120, 'BZ': 120,
    'UD': 130, 'GO': 130, 'PN': 130, 'TS': 130,
    'CH': 120, 'TE': 120, 'PE': 120, 'AQ': 110,
    'CB': 120, 'IS': 120,
    'PZ': 120, 'MT': 120,
    'CS': 120, 'CZ': 120, 'VV': 120, 'RC': 120,
    'CN': 130, 'AL': 130, 'AT': 130, 'VB': 130, 'SO': 120,
    'MI': 130, 'BG': 130, 'BS': 130, 'CO': 130, 'LC': 130, 'LO': 130, 'CR': 130, 'MN': 130, 'PV': 130,
})

def get_effective_circuity(sigla, base_circuity_dict):
    return base_circuity_dict.get(sigla, 1.35)

# =============================================================================
# FUNZIONI CORE
# =============================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

def calculate_leg_distance_v2(lon1, lat1, lon2, lat2, sigla_start):
    km = haversine_km(lon1, lat1, lon2, lat2)
    circuity = PROVINCIAL_CIRCUITY.get(sigla_start, 1.35)
    if km <= URBAN_RADIUS_KM:
        return km * circuity
    else:
        urban_km = URBAN_RADIUS_KM * circuity
        highway_km = (km - URBAN_RADIUS_KM) * HIGHWAY_CIRCUITY
        return urban_km + highway_km

def calculate_leg_time_v2(lon1, lat1, lon2, lat2, sigla_start):
    km = haversine_km(lon1, lat1, lon2, lat2)
    circuity = PROVINCIAL_CIRCUITY.get(sigla_start, 1.35)
    base_speed = PROVINCIAL_SPEED.get(sigla_start, 45)
    urban_speed = base_speed / circuity
    highway_speed = HIGHWAY_SPEED_BY_PROVINCE.get(sigla_start, 130)
    if km <= URBAN_RADIUS_KM:
        real_km = km * circuity
        return real_km / urban_speed
    else:
        urban_km = URBAN_RADIUS_KM * circuity
        highway_km = (km - URBAN_RADIUS_KM) * HIGHWAY_CIRCUITY
        time_urban = urban_km / urban_speed
        time_highway = highway_km / highway_speed
        return time_urban + time_highway

# =============================================================================
# CLASSIFICAZIONE 2 CLASSI (A / B)
# =============================================================================
def classify_abc(df, volume_col, threshold_a):
    df = df.copy()
    df['classe'] = np.where(df[volume_col] >= threshold_a, 'A', 'B')
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

def get_effective_speed(sigla, base_speed_dict):
    base_speed = base_speed_dict.get(sigla, 45)
    congestion = CONGESTION_FACTOR.get(sigla, 1.0)
    return base_speed / congestion

# =============================================================================
def _tour_2opt(lats, lons, siglas, circuity_dict, speed_dict, start_lat, start_lon, return_home=True):
    n = len(lats)
    if n == 0:
        return 0.0, 0.0, start_lat, start_lon, 0.0, 0.0, start_lat, start_lon, start_lat, start_lon
    remaining = set(range(n))
    tour = []
    cur_lat, cur_lon = start_lat, start_lon
    while remaining:
        best_idx = min(remaining, key=lambda idx: haversine_km(cur_lon, cur_lat, lons[idx], lats[idx]))
        tour.append(best_idx)
        remaining.remove(best_idx)
        cur_lat = lats[best_idx]
        cur_lon = lons[best_idx]
    if n >= 5:
        def _tour_dist(tour_idx):
            if not tour_idx: return 0.0
            total = calculate_leg_distance_v2(start_lon, start_lat, lons[tour_idx[0]], lats[tour_idx[0]], siglas[tour_idx[0]])
            for i in range(len(tour_idx) - 1):
                a, b = tour_idx[i], tour_idx[i+1]
                total += calculate_leg_distance_v2(lons[a], lats[a], lons[b], lats[b], siglas[a])
            if return_home:
                total += calculate_leg_distance_v2(lons[tour_idx[-1]], lats[tour_idx[-1]], start_lon, start_lat, siglas[tour_idx[-1]])
            return total
        def _swap(tour_idx, i, k):
            return tour_idx[:i] + tour_idx[i:k+1][::-1] + tour_idx[k+1:]
        current_tour = tour
        current_dist = _tour_dist(current_tour)
        for _ in range(min(50, n * n)):
            improved = False
            for i in range(n - 1):
                for k in range(i + 1, n):
                    new_tour = _swap(current_tour, i, k)
                    new_dist = _tour_dist(new_tour)
                    if new_dist < current_dist - 0.01:
                        current_tour = new_tour
                        current_dist = new_dist
                        improved = True
                        break
                if improved: break
            if not improved: break
        tour = current_tour
    first_idx, last_idx = tour[0], tour[-1]
    km_andata = calculate_leg_distance_v2(start_lon, start_lat, lons[first_idx], lats[first_idx], siglas[first_idx])
    km_intermediate = sum(
        calculate_leg_distance_v2(lons[tour[i]], lats[tour[i]], lons[tour[i+1]], lats[tour[i+1]], siglas[tour[i]])
        for i in range(len(tour) - 1)
    )
    km_ritorno = 0.0
    end_lat, end_lon = lats[last_idx], lons[last_idx]
    ore_viaggio = 0.0
    ore_viaggio += calculate_leg_time_v2(start_lon, start_lat, lons[first_idx], lats[first_idx], siglas[first_idx])
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i+1]
        ore_viaggio += calculate_leg_time_v2(lons[a], lats[a], lons[b], lats[b], siglas[a])
    if return_home:
        km_ritorno = calculate_leg_distance_v2(lons[last_idx], lats[last_idx], start_lon, start_lat, siglas[last_idx])
        ore_viaggio += calculate_leg_time_v2(lons[last_idx], lats[last_idx], start_lon, start_lat, siglas[last_idx])
        end_lat, end_lon = start_lat, start_lon
    tour_km = km_andata + km_intermediate + km_ritorno
    return (tour_km, ore_viaggio, end_lat, end_lon, km_andata, km_ritorno,
            lats[first_idx], lons[first_idx], lats[last_idx], lons[last_idx])


def create_geographic_daily_clusters(df_customers, rep_home_lat, rep_home_lon,
                                     max_stops_per_day, circuity_dict, speed_dict):
    if len(df_customers) == 0:
        return []
    df = df_customers.copy()
    total_visits = int(df['freq_visite'].sum())
    if total_visits == 0:
        return []
    visits = []
    for idx, row in df.iterrows():
        freq = int(row['freq_visite']) if pd.notna(row['freq_visite']) else 0
        for _ in range(freq):
            visits.append({
                'lat': row['latitudine'], 'lon': row['longitudine'], 'sigla': row['sigla'],
                'dist_home': haversine_km(row['longitudine'], row['latitudine'], rep_home_lon, rep_home_lat)
            })
    if not visits:
        return []
    visits_df = pd.DataFrame(visits)
    n_visits = len(visits_df)
    if n_visits <= max_stops_per_day:
        km, ore, end_lat, end_lon, km_andata, km_ritorno, f_lat, f_lon, l_lat, l_lon = _tour_2opt(
            visits_df['lat'].values, visits_df['lon'].values, visits_df['sigla'].values,
            circuity_dict, speed_dict, rep_home_lat, rep_home_lon, return_home=True
        )
        return [{
            'sector': 0, 'n_visite': n_visits, 'km': km, 'ore_viaggio': ore,
            'km_andata': km_andata, 'km_ritorno': km_ritorno, 'km_tour': km - km_andata - km_ritorno,
            'first_lat': f_lat, 'first_lon': f_lon, 'last_lat': l_lat, 'last_lon': l_lon,
            'sigle': visits_df['sigla'].tolist(),
            'clienti': [{'lat': r['lat'], 'lon': r['lon'], 'sigla': r['sigla']} for _, r in visits_df.iterrows()]
        }]
    visits_df = visits_df.sort_values('dist_home', ascending=False).reset_index(drop=True)
    coords = visits_df[['lat', 'lon']].values
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(coords[i, 1], coords[i, 0], coords[j, 1], coords[j, 0])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    assigned = np.zeros(n, dtype=bool)
    giornate = []
    while not assigned.all():
        unassigned_idx = np.where(~assigned)[0]
        if len(unassigned_idx) == 0:
            break
        seed_idx = unassigned_idx[0]
        cluster_indices = [seed_idx]
        assigned[seed_idx] = True
        while len(cluster_indices) < max_stops_per_day:
            remaining = np.where(~assigned)[0]
            if len(remaining) == 0:
                break
            min_dists = np.min(dist_matrix[np.ix_(remaining, cluster_indices)], axis=1)
            nearest_idx = remaining[np.argmin(min_dists)]
            cluster_indices.append(nearest_idx)
            assigned[nearest_idx] = True
        cluster_df = visits_df.iloc[cluster_indices]
        km, ore, end_lat, end_lon, km_andata, km_ritorno, f_lat, f_lon, l_lat, l_lon = _tour_2opt(
            cluster_df['lat'].values, cluster_df['lon'].values, cluster_df['sigla'].values,
            circuity_dict, speed_dict, rep_home_lat, rep_home_lon, return_home=True
        )
        giornate.append({
            'sector': len(giornate), 'n_visite': len(cluster_df), 'km': km, 'ore_viaggio': ore,
            'km_andata': km_andata, 'km_ritorno': km_ritorno, 'km_tour': km - km_andata - km_ritorno,
            'first_lat': f_lat, 'first_lon': f_lon, 'last_lat': l_lat, 'last_lon': l_lon,
            'sigle': cluster_df['sigla'].tolist(),
            'clienti': [{'lat': r['lat'], 'lon': r['lon'], 'sigla': r['sigla']} for _, r in cluster_df.iterrows()]
        })
    return giornate


def optimize_pernotto_v2(giornate, home_lat, home_lon, soglia_min,
                        max_notti_week, max_notti_consecutive, hotel_offset_km,
                        gg_lavoro, circuity_dict, speed_dict,
                        anti_schizofrenia_factor=0.75):
    n = len(giornate)
    if n < 2:
        return 0.0, 0.0, []
    all_speeds = [speed_dict.get(s, 45) for g in giornate for s in g.get('sigle', [])]
    avg_speed = np.mean(all_speeds) if all_speeds else 45
    soglia_km = (soglia_min / 60.0) * avg_speed
    max_notti_anno = max(1, int((gg_lavoro / 7.0) * max_notti_week))
    for g in giornate:
        clienti = g.get('clienti', [])
        if len(clienti) > 0:
            g['hotel_lat'] = np.mean([c['lat'] for c in clienti])
            g['hotel_lon'] = np.mean([c['lon'] for c in clienti])
        else:
            g['hotel_lat'], g['hotel_lon'] = home_lat, home_lon
        g['dist_da_casa'] = haversine_km(g['hotel_lon'], g['hotel_lat'], home_lon, home_lat)
    giornate.sort(key=lambda x: x['dist_da_casa'])
    coppie = []
    for i in range(n - 1):
        g_i, g_j = giornate[i], giornate[i + 1]
        sig_i = g_i['sigle'][-1] if g_i.get('sigle') else 'RM'
        sig_j = g_j['sigle'][0] if g_j.get('sigle') else 'RM'
        costo_attuale = (
            calculate_leg_distance_v2(g_i['last_lon'], g_i['last_lat'], home_lon, home_lat, sig_i) +
            calculate_leg_distance_v2(home_lon, home_lat, g_j['first_lon'], g_j['first_lat'], sig_j)
        )
        dist_to_hotel = calculate_leg_distance_v2(g_i['last_lon'], g_i['last_lat'], g_j['hotel_lon'], g_j['hotel_lat'], sig_i)
        costo_nuovo = dist_to_hotel + hotel_offset_km
        risparmio = costo_attuale - costo_nuovo
        troppo_vicino = g_j['dist_da_casa'] < (anti_schizofrenia_factor * soglia_km)
        if risparmio > soglia_km and not troppo_vicino:
            coppie.append((i, risparmio, dist_to_hotel, costo_attuale, costo_nuovo, troppo_vicino))
    coppie.sort(key=lambda x: x[1], reverse=True)
    used = [False] * n
    notti_usate = 0
    missioni = []
    for i, risp, dist_h, costo_att, costo_nuovo, troppo_vicino in coppie:
        if notti_usate >= max_notti_anno or used[i] or used[i + 1]:
            continue
        best_mission = (i, i + 1, risp, 1)
        used_mission = False
        if max_notti_consecutive >= 2 and (i + 2) < n and not used[i + 2]:
            g_i, g_i1, g_i2 = giornate[i], giornate[i + 1], giornate[i + 2]
            sig_i = g_i['sigle'][-1] if g_i.get('sigle') else 'RM'
            sig_i1 = g_i1['sigle'][0] if g_i1.get('sigle') else 'RM'
            sig_i2 = g_i2['sigle'][0] if g_i2.get('sigle') else 'RM'
            dist1 = calculate_leg_distance_v2(g_i['last_lon'], g_i['last_lat'], g_i1['hotel_lon'], g_i1['hotel_lat'], sig_i)
            risp1 = (calculate_leg_distance_v2(g_i['last_lon'], g_i['last_lat'], home_lon, home_lat, sig_i) +
                     calculate_leg_distance_v2(home_lon, home_lat, g_i1['first_lon'], g_i1['first_lat'], sig_i1)) - (dist1 + hotel_offset_km)
            dist2 = calculate_leg_distance_v2(g_i1['last_lon'], g_i1['last_lat'], g_i2['hotel_lon'], g_i2['hotel_lat'], sig_i1)
            risp2 = (calculate_leg_distance_v2(g_i1['last_lon'], g_i1['last_lat'], home_lon, home_lat, sig_i1) +
                     calculate_leg_distance_v2(home_lon, home_lat, g_i2['first_lon'], g_i2['first_lat'], sig_i2)) - (dist2 + hotel_offset_km)
            risp_tot = risp1 + risp2
            troppo_vicino_g2 = g_i2['dist_da_casa'] < (anti_schizofrenia_factor * soglia_km)
            if risp_tot > 1.5 * soglia_km * 2 and not troppo_vicino_g2 and notti_usate + 2 <= max_notti_anno:
                best_mission = (i, i + 2, risp_tot, 2)
                used[i] = used[i + 1] = used[i + 2] = True
                notti_usate += 2
                used_mission = True
        if not used_mission:
            used[i] = used[i + 1] = True
            notti_usate += 1
        missioni.append(best_mission)
    total_delta_km, total_delta_ore, dettaglio = 0.0, 0.0, []
    for start, end, risp_tot, n_notti in missioni:
        for idx in range(start, end):
            g_curr, g_next = giornate[idx], giornate[idx + 1]
            sig_curr = g_curr['sigle'][-1] if g_curr.get('sigle') else 'RM'
            sig_next = g_next['sigle'][0] if g_next.get('sigle') else 'RM'
            delta_km = -calculate_leg_distance_v2(g_curr['last_lon'], g_curr['last_lat'], home_lon, home_lat, sig_curr)
            dist_to_hotel = calculate_leg_distance_v2(g_curr['last_lon'], g_curr['last_lat'], g_next['hotel_lon'], g_next['hotel_lat'], sig_curr)
            delta_km += dist_to_hotel
            delta_km += -calculate_leg_distance_v2(home_lon, home_lat, g_next['first_lon'], g_next['first_lat'], sig_next)
            delta_km += hotel_offset_km
            time_ritorno = calculate_leg_time_v2(g_curr['last_lon'], g_curr['last_lat'], home_lon, home_lat, sig_curr)
            time_to_hotel = calculate_leg_time_v2(g_curr['last_lon'], g_curr['last_lat'], g_next['hotel_lon'], g_next['hotel_lat'], sig_curr)
            time_andata_next = calculate_leg_time_v2(home_lon, home_lat, g_next['first_lon'], g_next['first_lat'], sig_next)
            time_hotel_offset = hotel_offset_km / HIGHWAY_SPEED_BY_PROVINCE.get(sig_next, 130)
            delta_ore = -time_ritorno + time_to_hotel - time_andata_next + time_hotel_offset
            total_delta_km += delta_km
            total_delta_ore += delta_ore
            dettaglio.append({
                'giorno_da': idx, 'giorno_a': idx + 1, 'delta_km': round(delta_km, 2), 'delta_ore': round(delta_ore, 2),
                'hotel_lat': g_next['hotel_lat'], 'hotel_lon': g_next['hotel_lon'],
                'notti_in_missione': n_notti if idx == start else 0
            })
    return total_delta_km, total_delta_ore, dettaglio


def generate_weekly_calendar(df_customers, col_vol, threshold_a, freq_a, freq_b, gg_lavoro):
    n_settimane = max(1, gg_lavoro // 5)
    df = classify_abc(df_customers.copy(), col_vol, threshold_a)
    freq_map = {'A': freq_a, 'B': freq_b}
    df['freq_visite'] = df['classe'].map(freq_map).fillna(0).astype(int)
    calendar = {}
    for rep in df['sales rep'].unique():
        if pd.isna(rep):
            continue
        rep_clients = df[df['sales rep'] == rep].copy()
        calendar[rep] = {w: [] for w in range(1, n_settimane + 1)}
        for idx, row in rep_clients.iterrows():
            freq = int(row['freq_visite'])
            if freq <= 0:
                continue
            if freq >= n_settimane:
                weeks = list(range(1, n_settimane + 1))
            else:
                step = n_settimane / freq
                weeks = []
                for i in range(freq):
                    w = int(round((i + 0.5) * step))
                    w = max(1, min(n_settimane, w))
                    weeks.append(w)
                weeks = sorted(list(set(weeks)))
            for w in weeks:
                calendar[rep][w].append(idx)
    return calendar


def build_weekly_tour(rep, week_indices, df_customers,
                      home_lat, home_lon,
                      max_stops_per_day, circuity_dict, speed_dict,
                      pernotto_attivo, soglia_min_pernotto,
                      max_notti_week, max_notti_consecutive,
                      hotel_offset_km, pausa_km, pausa_min_gg):
    if not week_indices:
        return 0.0, 0.0, [], 0
    sub = df_customers.loc[list(set(week_indices))].copy()
    if len(sub) == 0:
        return 0.0, 0.0, [], 0
    sub['dist_home'] = sub.apply(lambda r: haversine_km(r['longitudine'], r['latitudine'], home_lon, home_lat), axis=1)
    sub = sub.sort_values('dist_home', ascending=False)
    clienti_rimasti = sub.to_dict('records')
    giornate = []
    while clienti_rimasti:
        obiettivo = clienti_rimasti.pop(0)
        giornata_clienti = [obiettivo]
        while len(giornata_clienti) < max_stops_per_day and clienti_rimasti:
            last_lat = giornata_clienti[-1]['latitudine']
            last_lon = giornata_clienti[-1]['longitudine']
            best_idx = -1
            best_dist = float('inf')
            for i, c in enumerate(clienti_rimasti):
                d = haversine_km(last_lon, last_lat, c['longitudine'], c['latitudine'])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0:
                giornata_clienti.append(clienti_rimasti.pop(best_idx))
        lats = np.array([c['latitudine'] for c in giornata_clienti])
        lons = np.array([c['longitudine'] for c in giornata_clienti])
        siglas = [c['sigla'] for c in giornata_clienti]
        km, ore, end_lat, end_lon, km_andata, km_ritorno, f_lat, f_lon, l_lat, l_lon = _tour_2opt(
            lats, lons, siglas, circuity_dict, speed_dict, home_lat, home_lon, return_home=True
        )
        giornate.append({
            'km': km, 'ore_viaggio': ore, 'km_andata': km_andata, 'km_ritorno': km_ritorno,
            'end_lat': end_lat, 'end_lon': end_lon, 'clienti': giornata_clienti,
            'n_visite': len(giornata_clienti), 'sigle': siglas,
            'first_lon': f_lon, 'first_lat': f_lat, 'last_lon': l_lon, 'last_lat': l_lat,
        })
    total_delta_km = 0.0
    total_delta_ore = 0.0
    notti = 0
    consecutive = 0
    if pernotto_attivo and len(giornate) >= 2:
        all_speeds = [speed_dict.get(c['sigla'], 45) for g in giornate for c in g['clienti']]
        avg_speed = np.mean(all_speeds) if all_speeds else 45
        soglia_km = (soglia_min_pernotto / 60.0) * avg_speed
        for i in range(len(giornate) - 1):
            g_i = giornate[i]
            g_j = giornate[i + 1]
            hotel_lat = np.mean([c['latitudine'] for c in g_j['clienti']])
            hotel_lon = np.mean([c['longitudine'] for c in g_j['clienti']])
            sig_i = g_i['sigle'][-1] if g_i.get('sigle') else 'RM'
            sig_j = g_j['sigle'][0] if g_j.get('sigle') else 'RM'
            costo_attuale = (
                calculate_leg_distance_v2(g_i['last_lon'], g_i['last_lat'], home_lon, home_lat, sig_i) +
                calculate_leg_distance_v2(home_lon, home_lat, g_j['first_lon'], g_j['first_lat'], sig_j)
            )
            dist_to_hotel = calculate_leg_distance_v2(g_i['end_lon'], g_i['end_lat'], hotel_lon, hotel_lat, sig_i)
            costo_nuovo = dist_to_hotel + hotel_offset_km
            risparmio = costo_attuale - costo_nuovo
            dist_gj_da_casa = haversine_km(hotel_lon, hotel_lat, home_lon, home_lat)
            troppo_vicino = dist_gj_da_casa < (0.75 * soglia_km)
            if (risparmio > soglia_km and not troppo_vicino and
                notti < max_notti_week and consecutive < max_notti_consecutive):
                time_ritorno_i = calculate_leg_time_v2(g_i['last_lon'], g_i['last_lat'], home_lon, home_lat, sig_i)
                time_to_hotel = calculate_leg_time_v2(g_i['end_lon'], g_i['end_lat'], hotel_lon, hotel_lat, sig_i)
                time_andata_j = calculate_leg_time_v2(home_lon, home_lat, g_j['first_lon'], g_j['first_lat'], sig_j)
                time_hotel_offset = hotel_offset_km / HIGHWAY_SPEED_BY_PROVINCE.get(sig_j, 130)
                delta_km = -calculate_leg_distance_v2(g_i['last_lon'], g_i['last_lat'], home_lon, home_lat, sig_i)
                delta_km += dist_to_hotel
                delta_km += -calculate_leg_distance_v2(home_lon, home_lat, g_j['first_lon'], g_j['first_lat'], sig_j)
                delta_km += hotel_offset_km
                delta_ore = -time_ritorno_i + time_to_hotel - time_andata_j + time_hotel_offset
                total_delta_km += delta_km
                total_delta_ore += delta_ore
                notti += 1
                consecutive += 1
                g_i['km'] += delta_km
                g_i['ore_viaggio'] += delta_ore
                g_i['km_ritorno'] = 0
                g_j['km_andata'] = hotel_offset_km
            else:
                consecutive = 0
    total_km = sum(g['km'] for g in giornate) + total_delta_km + (pausa_km * len(giornate))
    total_ore = sum(g['ore_viaggio'] for g in giornate) + total_delta_ore + ((pausa_min_gg / 60.0) * len(giornate))
    return total_km, total_ore, giornate, notti


def calculate_travel_km_tours_v2(df_customers, rep_home_lat, rep_home_lon,
                                max_stops_per_day, circuity_dict, speed_dict,
                                pernotto_attivo=False, soglia_min_pernotto=120,
                                max_notti_week=1, max_notti_consecutive=2,
                                hotel_offset_km=10, gg_lavoro=220,
                                anti_schizofrenia_factor=0.75,
                                pausa_km=15.0, pausa_min_gg=20,
                                col_vol='volume_calcolato',
                                threshold_a=700,
                                freq_a=20, freq_b=12):
    if len(df_customers) == 0:
        return 0.0, 0.0, [], 0
    n_settimane = max(1, gg_lavoro // 5)
    df = classify_abc(df_customers.copy(), col_vol, threshold_a)
    freq_map = {'A': freq_a, 'B': freq_b}
    df['freq_visite'] = df['classe'].map(freq_map).fillna(0).astype(int)
    calendar = {w: [] for w in range(1, n_settimane + 1)}
    for idx, row in df.iterrows():
        freq = int(row['freq_visite'])
        if freq <= 0:
            continue
        if freq >= n_settimane:
            weeks = list(range(1, n_settimane + 1))
        else:
            step = n_settimane / freq
            weeks = []
            for i in range(freq):
                w = int(round((i + 0.5) * step))
                w = max(1, min(n_settimane, w))
                weeks.append(w)
            weeks = sorted(list(set(weeks)))
        for w in weeks:
            calendar[w].append(idx)
    total_km = 0.0
    total_ore = 0.0
    all_giornate = []
    total_pernotti = 0
    for w in range(1, n_settimane + 1):
        week_indices = calendar[w]
        if not week_indices:
            continue
        km_w, ore_w, giornate_w, pernotti_w = build_weekly_tour(
            None, week_indices, df_customers, rep_home_lat, rep_home_lon,
            max_stops_per_day, circuity_dict, speed_dict,
            pernotto_attivo, soglia_min_pernotto,
            max_notti_week, max_notti_consecutive,
            hotel_offset_km, pausa_km, pausa_min_gg
        )
        total_km += km_w
        total_ore += ore_w
        all_giornate.extend(giornate_w)
        total_pernotti += pernotti_w
    return total_km, total_ore, all_giornate, total_pernotti


def compute_alpha_shape(df_customers, alpha_factor=0.15):
    try:
        from shapely.geometry import MultiPoint
        pts = df_customers[['longitudine', 'latitudine']].values
        if len(pts) < 3:
            return None, None
        multipoint = MultiPoint(pts)
        hull = multipoint.convex_hull
        bounds = hull.bounds
        size = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        buffer_size = -size * alpha_factor
        if buffer_size >= 0:
            return None, None
        concave = hull.buffer(buffer_size)
        if concave.is_empty:
            return None, None
        if concave.geom_type == 'Polygon':
            x, y = concave.exterior.xy
            return list(x), list(y)
        elif concave.geom_type == 'MultiPolygon':
            largest = max(concave.geoms, key=lambda p: p.area)
            x, y = largest.exterior.xy
            return list(x), list(y)
    except Exception:
        return None, None


def build_territory_map(df_work, df_v, active_reps, title=""):
    df_map = df_work.dropna(subset=['latitudine', 'longitudine', 'assigned_rep'])
    fig = go.Figure()
    if len(df_map) == 0:
        return fig
    def hsv_palette(n):
        pal = []
        for i in range(n):
            h = (i * 0.618033988749895) % 1.0
            s = 0.75 + 0.25 * ((i % 3) / 2.0)
            v = 0.85 + 0.15 * ((i % 2))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pal.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
        return pal
    colors = hsv_palette(len(active_reps))
    rep_colors = {r: colors[i] for i, r in enumerate(active_reps)}
    for rep in active_reps:
        sub = df_map[df_map['assigned_rep'] == rep]
        if len(sub) >= 3:
            lon_h, lat_h = compute_alpha_shape(sub)
            if lon_h is not None:
                fig.add_trace(go.Scattermapbox(
                    mode='lines', lon=lon_h, lat=lat_h,
                    line=dict(width=2, color=rep_colors[rep]),
                    fill='toself', fillcolor=rep_colors[rep], opacity=0.12,
                    name=f"Zona {rep}", hoverinfo='name'
                ))
            else:
                lon_h, lat_h = compute_hull_coords(sub)
                if lon_h is not None:
                    fig.add_trace(go.Scattermapbox(
                        mode='lines', lon=lon_h, lat=lat_h,
                        line=dict(width=1.5, color=rep_colors[rep]),
                        fill='toself', fillcolor=rep_colors[rep], opacity=0.10,
                        name=f"Zona {rep}", hoverinfo='name'
                    ))
    classe_colors = {'A': '#e53935', 'B': '#1e88e5'}
    classe_sizes = {'A': 9, 'B': 4}
    for classe in ['A', 'B']:
        df_cl = df_map[df_map['classe'] == classe]
        if len(df_cl) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=df_cl['latitudine'], lon=df_cl['longitudine'],
                mode='markers',
                marker=dict(size=classe_sizes[classe], color=classe_colors[classe], opacity=0.85),
                name=f"Class {classe}",
                text=df_cl['assigned_rep'].values,
                hoverinfo='name+text'
            ))
    df_v_active = df_v[df_v['sales rep'].isin(active_reps)].dropna(subset=['latitudine', 'longitudine'])
    if len(df_v_active) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=df_v_active['latitudine'], lon=df_v_active['longitudine'],
            mode='markers+text',
            marker=dict(size=16, color='black', opacity=0.9),
            text=df_v_active['sales rep'].values,
            textposition='top right', textfont=dict(size=10, color='black'),
            name='🏠 Home Base', hoverinfo='name'
        ))
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=5.5,
        mapbox_center=dict(lat=42.5, lon=12.5),
        margin=dict(r=0, t=30, l=0, b=120), height=700, title=title,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                   bgcolor='rgba(255,255,255,0.95)', bordercolor='gray', borderwidth=1)
    )
    return fig


def _run_simulation_cached(df_c_json, df_v_json, active_list, col_vol, threshold_a,
                          freq_a, freq_b, dur_visita, ore_gg, gg_lavoro,
                          max_stops_per_day, pernotto_attivo, soglia_min_pernotto,
                          max_notti_week, max_notti_consecutive, hotel_offset_km,
                          pausa_km=15.0, pausa_min_gg=20):
    df_c = pd.read_json(StringIO(df_c_json))
    df_v = pd.read_json(StringIO(df_v_json))
    return run_simulation(df_c, df_v, active_list, col_vol, threshold_a,
                         freq_a, freq_b, dur_visita, ore_gg, gg_lavoro,
                         max_stops_per_day, pernotto_attivo, soglia_min_pernotto,
                         max_notti_week, max_notti_consecutive, hotel_offset_km,
                         pausa_km, pausa_min_gg)

def run_simulation(df_c, df_v, active_list, col_vol, threshold_a,
                  freq_a, freq_b, dur_visita, ore_gg, gg_lavoro,
                  max_stops_per_day,
                  pernotto_attivo=False, soglia_min_pernotto=120,
                  max_notti_week=1, max_notti_consecutive=2,
                  hotel_offset_km=10,
                  pausa_km=15.0, pausa_min_gg=20):
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

    if 'sales rep' not in df_w.columns:
        return None, "Errore: colonna 'sales rep' mancante nel foglio clienti."

    df_w['assigned_rep'] = df_w['sales rep']
    df_w = df_w[df_w['assigned_rep'].isin(active_list)].copy()

    if len(df_w) == 0:
        return None, "Errore: nessun cliente corrisponde ai venditori attivi. Verifica che i nomi 'sales rep' nei fogli clienti e venditori corrispondano."

    df_w = df_w.merge(
        df_v_valid[['sales rep', 'latitudine', 'longitudine']].rename(
            columns={'latitudine': 'rep_lat', 'longitudine': 'rep_lon', 'sales rep': 'assigned_rep'}
        ),
        on='assigned_rep', how='left'
    )

    df_w['dist_km'] = haversine_km(
        df_w['longitudine'].values, df_w['latitudine'].values,
        df_w['rep_lon'].values, df_w['rep_lat'].values
    )
    df_w['assegnazione'] = 'attuale'

    # Classificazione 2 classi
    df_w = classify_abc(df_w, col_vol, threshold_a)

    # Aggregati completi per venditore
    agg_list = []
    for rep in active_list:
        sub = df_w[df_w['assigned_rep'] == rep]
        if len(sub) > 0:
            agg_list.append({
                'sales_rep': rep,
                'n_clienti': int(len(sub)),
                'n_classe_a': int((sub['classe'] == 'A').sum()),
                'n_classe_b': int((sub['classe'] == 'B').sum()),
                'volume_a': float(sub.loc[sub['classe'] == 'A', col_vol].sum()),
                'volume_b': float(sub.loc[sub['classe'] == 'B', col_vol].sum()),
            })
        else:
            agg_list.append({
                'sales_rep': rep, 'n_clienti': 0,
                'n_classe_a': 0, 'n_classe_b': 0,
                'volume_a': 0.0, 'volume_b': 0.0,
            })

    agg = pd.DataFrame(agg_list)

    # Tutti i clienti sono attivi (non c'è più classe D)
    df_work = df_w.copy()

    # Coordinate venditore su df_work
    df_work['rep_lat'] = df_work['assigned_rep'].map(df_v_valid.set_index('sales rep')['latitudine'])
    df_work['rep_lon'] = df_work['assigned_rep'].map(df_v_valid.set_index('sales rep')['longitudine'])

    freq_map = {'A': freq_a, 'B': freq_b}
    df_work['freq_visite'] = df_work['classe'].map(freq_map)
    df_work['ore_visita_annue'] = (df_work['freq_visite'] * dur_visita) / 60.0

    # Calcolo km e ore viaggio
    travel_data = []
    for rep in active_list:
        sub = df_work[df_work['assigned_rep'] == rep]
        if len(sub) > 0:
            rep_lat = sub['rep_lat'].iloc[0]
            rep_lon = sub['rep_lon'].iloc[0]
            km_totali, ore_viag, _, n_pernotti = calculate_travel_km_tours_v2(
                sub, rep_lat, rep_lon, max_stops_per_day,
                PROVINCIAL_CIRCUITY, PROVINCIAL_SPEED,
                pernotto_attivo, soglia_min_pernotto,
                max_notti_week, max_notti_consecutive,
                hotel_offset_km, gg_lavoro,
                pausa_km=pausa_km, pausa_min_gg=pausa_min_gg,
                col_vol=col_vol,
                threshold_a=threshold_a,
                freq_a=freq_a, freq_b=freq_b
            )
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': ore_viag, 'km_annui': km_totali, 'n_pernotti_annui': n_pernotti})
        else:
            travel_data.append({'sales_rep': rep, 'ore_viaggio_annue': 0.0, 'km_annui': 0.0, 'n_pernotti_annui': 0})

    travel_df = pd.DataFrame(travel_data)
    agg = agg.merge(travel_df, on='sales_rep', how='left').fillna(0)

    agg['km_giorno'] = agg['km_annui'] / gg_lavoro

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
    df_v_valid = df_v.dropna(subset=['latitudine', 'longitudine'])
    scores = []
    for rep in active_reps:
        rep_rows = df_v_valid[df_v_valid['sales rep'] == rep]
        if len(rep_rows) == 0:
            continue
        rep_clients = df_c[df_c['sales rep'] == rep].dropna(subset=['latitudine', 'longitudine'])
        if len(rep_clients) > 0:
            centroid_lat = rep_clients['latitudine'].mean()
            centroid_lon = rep_clients['longitudine'].mean()
        else:
            rep_row = rep_rows.iloc[0]
            centroid_lat = rep_row['latitudine']
            centroid_lon = rep_row['longitudine']
        dists = orphan_df.apply(
            lambda row: haversine_km(row['longitudine'], row['latitudine'], centroid_lon, centroid_lat),
            axis=1
        )
        avg_dist = dists.mean() if len(dists) > 0 else 9999
        if avg_dist > max_km:
            continue
        if current_result is not None and rep in current_result['sales_rep'].values:
            sat = current_result[current_result['sales_rep'] == rep]['saturazione_pct'].iloc[0]
        else:
            sat = 50.0
        if sat > 100:
            score = 0.0
            cap_residua = 0.0
        else:
            cap_residua = max(0.0, 100.0 - sat)
            dist_score = max(0.0, 1.0 - avg_dist / 300.0)
            sat_score = cap_residua / 100.0
            score = 0.5 * dist_score + 0.5 * sat_score
        scores.append((rep, score, avg_dist, sat, cap_residua))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def preview_allocation(orphan_df, selected_receivers, df_c, df_v):
    v_centroids = {}
    for r in selected_receivers:
        rep_clients = df_c[df_c['sales rep'] == r].dropna(subset=['latitudine', 'longitudine'])
        if len(rep_clients) > 0:
            v_centroids[r] = (rep_clients['latitudine'].mean(), rep_clients['longitudine'].mean())
        else:
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
            d = haversine_km(row['longitudine'], row['latitudine'], v_centroids[r][1], v_centroids[r][0])
            if d < best_d:
                best_d = d
                best_r = r
        if best_r:
            allocation[best_r].append((idx, best_d))
    return allocation

def apply_reassignment(df_c, removed_rep, selected_receivers, df_v):
    df = df_c.copy()
    orphan_mask = df['sales rep'] == removed_rep
    if not orphan_mask.any():
        return df
    orphan_df = df[orphan_mask].copy()
    v_centroids = {}
    for r in selected_receivers:
        rep_clients = df[df['sales rep'] == r].dropna(subset=['latitudine', 'longitudine'])
        if len(rep_clients) > 0:
            v_centroids[r] = (rep_clients['latitudine'].mean(), rep_clients['longitudine'].mean())
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
            d = haversine_km(row['longitudine'], row['latitudine'], v_centroids[r][1], v_centroids[r][0])
            if d < best_d:
                best_d = d
                best_r = r
        if best_r:
            df.loc[idx, 'sales rep'] = best_r
    return df

def render_html_table(headers, rows, font_size="12px"):
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
def generate_reassignment_map(orphan_df, allocation, df_v, removed_rep, receivers, title=""):
    fig = go.Figure()
    removed_home = df_v[df_v['sales rep'] == removed_rep].dropna(subset=['latitudine', 'longitudine'])
    if len(removed_home) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=removed_home['latitudine'].tolist(), lon=removed_home['longitudine'].tolist(),
            mode='markers', marker=dict(size=20, color='black', symbol='x', opacity=0.9),
            name=f'❌ {removed_rep} (rimosso)'
        ))
    colors = px.colors.qualitative.Bold
    for i, receiver in enumerate(receivers):
        color = colors[i % len(colors)]
        recv_home = df_v[df_v['sales rep'] == receiver].dropna(subset=['latitudine', 'longitudine'])
        if len(recv_home) == 0:
            continue
        recv_lat = recv_home.iloc[0]['latitudine']
        recv_lon = recv_home.iloc[0]['longitudine']
        fig.add_trace(go.Scattermapbox(
            lat=[recv_lat], lon=[recv_lon], mode='markers',
            marker=dict(size=14, color=color, opacity=0.9), name=f'🏠 {receiver}'
        ))
        if receiver in allocation and len(allocation[receiver]) > 0:
            assigned_indices = [idx for idx, _ in allocation[receiver]]
            sub = orphan_df.loc[assigned_indices]
            for idx, row in sub.iterrows():
                fig.add_trace(go.Scattermapbox(
                    mode='lines', lat=[row['latitudine'], recv_lat], lon=[row['longitudine'], recv_lon],
                    line=dict(width=1.5, color=color), opacity=0.5, showlegend=False, hoverinfo='skip'
                ))
            hover_text = []
            for idx, row in sub.iterrows():
                nome = row.get('ragione sociale', row.get('cliente', f'Cliente {idx}'))
                hover_text.append(f"{nome}<br>→ {receiver}")
            fig.add_trace(go.Scattermapbox(
                lat=sub['latitudine'], lon=sub['longitudine'], mode='markers',
                marker=dict(size=9, color=color, opacity=0.9),
                name=f'Clienti → {receiver}', text=hover_text, hoverinfo='text'
            ))
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=5.5,
        mapbox_center=dict(lat=42.5, lon=12.5),
        margin=dict(r=0, t=40, l=0, b=120), height=700,
        title=title or f"Riassegnazione: {removed_rep}",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                   bgcolor='rgba(255,255,255,0.95)', bordercolor='gray', borderwidth=1)
    )
    return fig

def fig_to_png(fig):
    try:
        import plotly.io as pio
        img_bytes = pio.to_image(fig, format="png", width=1600, height=1000, scale=2)
        return img_bytes
    except Exception as e:
        st.warning(f"⚠️ Export PNG fallito (serve `pip install kaleido`): {e}")
        return None

def generate_excel_report(initial_result, final_result, reassignment_history, removal_details, col_vol):
    if initial_result is None or final_result is None:
        return None
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        init_sheet = initial_result.copy()
        init_sheet.to_excel(writer, sheet_name='Stato Iniziale', index=False)
        final_sheet = final_result.copy()
        final_sheet.to_excel(writer, sheet_name='Stato Finale', index=False)
        all_reps = sorted(set(init_sheet['sales_rep'].tolist() + final_sheet['sales_rep'].tolist()))
        metrics = ['n_clienti', 'n_classe_a', 'n_classe_b',
                  'volume_a', 'volume_b', 'ore_visite_annue', 'ore_viaggio_annue',
                  'ore_totali_annue', 'saturazione_pct', 'km_annui', 'driving_min_giorno', 'visite_giorno']
        metric_labels = ['Customers', 'Class A', 'Class B',
                        'Volumes A', 'Volumes B', 'Visits Time', 'Driving Time',
                        'Total Time', 'Time Capacity %', 'KM Annui', 'Min/Day', 'Visit/Day']
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
                                'Step_Rimozione': removed, 'Ricevente': receiver,
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
        if reassignment_history:
            pd.DataFrame(reassignment_history).to_excel(writer, sheet_name='Riepilogo Step', index=False)
    output.seek(0)
    return output


# =============================================================================
# INTERFACCIA
# =============================================================================
def main():
    # Inizializzazione stato sessione - abc_vals con default per evitare errori
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
    # FIX: inizializzazione diretta con valori default, non None
    if 'abc_vals' not in st.session_state or st.session_state.abc_vals is None:
        st.session_state.abc_vals = {
            'threshold_a': 700,
            'freq_a': 20,
            'freq_b': 12,
        }
    if 'min_vol' not in st.session_state: st.session_state.min_vol = 0
    if 'max_km_filter' not in st.session_state: st.session_state.max_km_filter = 200

    if 'initial_result' not in st.session_state: st.session_state.initial_result = None
    if 'initial_df_work' not in st.session_state: st.session_state.initial_df_work = None
    if 'initial_fig' not in st.session_state: st.session_state.initial_fig = None
    if 'removal_figures' not in st.session_state: st.session_state.removal_figures = {}
    if 'removal_details' not in st.session_state: st.session_state.removal_details = {}
    if 'final_fig' not in st.session_state: st.session_state.final_fig = None

    # Migrazione formato classificazione vecchio → nuovo
    if st.session_state.abc_vals is not None and isinstance(st.session_state.abc_vals, dict) and 'da_a' in st.session_state.abc_vals:
        st.session_state.abc_vals = {
            'threshold_a': st.session_state.abc_vals.get('da_a', 700),
            'freq_a': st.session_state.abc_vals.get('freq_a', 20),
            'freq_b': st.session_state.abc_vals.get('freq_b', 12),
        }

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
            st.session_state.initial_result = None
            st.session_state.initial_df_work = None
            st.session_state.initial_fig = None
            st.session_state.removal_figures = {}
            st.session_state.removal_details = {}
            st.session_state.final_fig = None
            # FIX: reset abc_vals ai default
            st.session_state.abc_vals = {
                'threshold_a': 700,
                'freq_a': 20,
                'freq_b': 12,
            }

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
        available_cols = vol_cols if vol_cols else df_c.columns.tolist()

        selected_vol_cols = st.multiselect(
            "📊 Colonne Volume (somma 1-5)",
            available_cols,
            default=available_cols[:1] if available_cols else [],
            max_selections=5,
            key="vol_cols_select"
        )

        if len(selected_vol_cols) == 0:
            st.error("❌ Seleziona almeno una colonna volume")
            st.stop()

        if len(selected_vol_cols) > 1:
            col_vol = 'volume_calcolato'
        else:
            col_vol = selected_vol_cols[0]

        # TABELLA VOLUMI BRAND SIDEBAR
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        vol_rows = []
        grand_total = 0.0
        for c in selected_vol_cols:
            tot = pd.to_numeric(df_c[c], errors='coerce').fillna(0).sum()
            grand_total += tot
            vol_rows.append([c, fmt_eu(tot, 0)])
        vol_rows.append(['<<b>TOTALE</b>', f"<b>{fmt_eu(grand_total, 0)}</b>"])

        st.markdown(
            "<style>.brand-table td,.brand-table th{padding:4px 8px;font-size:12px;text-align:center;}"
            ".brand-table th{background:#333;color:#fff;border-bottom:1px solid #555;}"
            ".brand-table td{border-bottom:1px solid #444;color:#fff;}"
            ".brand-table tr:last-child td{background:#2e7d32;color:#fff;font-weight:bold;}</style>"
            + render_html_table(['Brand', 'Volume'], vol_rows, "12px").replace('<table', '<table class="brand-table"'),
            unsafe_allow_html=True
        )

        dur_visita = st.slider("⏱️ Durata media visita (min)", 40, 150, 90, step=5)
        ore_gg = st.number_input("🕒 Ore lavorative/giorno", 6.0, 10.0, 8.0, step=0.5)
        gg_lavoro = st.number_input("📅 Giorni lavorativi/anno", 180, 260, 220, step=5)
        pausa_pranzo = st.slider("🍽️ Pausa pranzo (min/giorno)", 0, 120, 100, step=5)
        ore_effettive_gg = ore_gg - (pausa_pranzo / 60.0)
        st.caption(f"*Capacità annua: {ore_effettive_gg * gg_lavoro:,.0f} ore*")

        st.markdown("**🍽️ Spostamento pranzo**")
        pausa_km = st.number_input("Km giornalieri per pranzo", 0.0, 50.0, 15.0, step=5.0, key="pausa_km")
        pausa_min_gg = st.number_input("Minuti di guida per pranzo", 0, 60, 20, step=5, key="pausa_min")
        st.caption(f"*Aggiunge {pausa_km:.1f} km e {pausa_min_gg} min di guida/giorno*")
        max_stops_per_day = st.slider("📦 Max visite/giorno", 3, 10, 5, step=1)
        st.divider()

        st.subheader("🏨 Logica di Pernotto")
        pernotto_attivo = st.toggle("Attiva Pernotto", value=False, key="pernotto_toggle")

        if pernotto_attivo:
            st.markdown("<div style='background:#1e1e1e;padding:12px;border-radius:8px;border-left:4px solid #4caf50;'>"
                       "<b>🟢 Pernotto attivo</b><br>"
                       "Il venditore non torna a casa se il risparmio supera la soglia. "
                       "Il giorno successivo parte dal centroide dei clienti del giorno dopo."
                       "</div>", unsafe_allow_html=True)
            soglia_min_pernotto = st.slider("Soglia min. guida (min) per attivare pernotto", 60, 240, 120, step=10)
            max_notti_week = st.slider("Max notti/settimana", 0, 3, 1, step=1)
            max_notti_consecutive = st.slider("Max notti consecutive", 1, 3, 2, step=1,
                                            help="2 notti = missione 3 giorni")
            hotel_offset_km = st.number_input("Distanza hotel → primo cliente (km)", 0.0, 50.0, 10.0, step=5.0,
                                            help="Stima del viaggio mattutino hotel-primo cliente")
        else:
            soglia_min_pernotto = 120
            max_notti_week = 1
            max_notti_consecutive = 2
            hotel_offset_km = 10.0

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
                    reset_key = f"reset_rep_{r}"
                    if reset_key in st.session_state and st.session_state[reset_key]:
                        default_val = True
                        del st.session_state[reset_key]
                    else:
                        default_val = st.session_state.rep_status_prev.get(r, True)
                    rep_status[r] = st.checkbox(r, value=default_val, key=f"rep_{r}")

    # =============================================================================
    # DOPO SIDEBAR
    # =============================================================================
    df_c = st.session_state.df_c_working

    if 'selected_vol_cols' in locals() and len(selected_vol_cols) > 1:
        df_c['volume_calcolato'] = df_c[selected_vol_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

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
                    st.session_state[f"reset_rep_{removed_name}"] = True
                    st.session_state.pending_removal = None
                    st.rerun()
            st.stop()

        # Classifica orfani
        abc = st.session_state.abc_vals
        orphan_df = classify_abc(orphan_df, col_vol, abc['threshold_a'])

        n_a = int((orphan_df['classe'] == 'A').sum())
        n_b = int((orphan_df['classe'] == 'B').sum())
        vol_a = float(orphan_df.loc[orphan_df['classe'] == 'A', col_vol].sum()) if n_a > 0 else 0.0
        vol_b = float(orphan_df.loc[orphan_df['classe'] == 'B', col_vol].sum()) if n_b > 0 else 0.0

        st.markdown(f"### 📍 {n_orfani} clienti orfani da {removed_name}")

        orphan_headers = ['Metrica', '🟢 Class A', '🔴 Class B', '🔵 Totali']
        orphan_rows = [
            ['N. clienti', str(n_a), str(n_b), f'**{n_orfani}**'],
            ['Volume', fmt_eu(vol_a), fmt_eu(vol_b), f'**{fmt_eu(vol_a + vol_b)}**']
        ]
        st.markdown(render_html_table(orphan_headers, orphan_rows, "13px"), unsafe_allow_html=True)
        st.divider()

        st.markdown("### 🎯 Seleziona i riceventi")
        active_reps = [r for r in reps if r != removed_name and r not in removed_reps]

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

        recv_data = []
        for i, (rep, score, avg_dist, sat, cap) in enumerate(ranked):
            is_suggested = i < 3
            rep_current_clients = st.session_state.df_c_working[
                st.session_state.df_c_working['sales rep'] == rep
            ].copy()
            rep_current_clients = classify_abc(rep_current_clients, col_vol, abc['threshold_a'])
            current_abc = len(rep_current_clients)
            recv_data.append({
                'Seleziona': is_suggested,
                'Ricevente': rep,
                'Clienti A+B attuali': current_abc,
                'Sat. Attuale': sat_emoji(sat),
                'Distanza media (km) nuovi clienti': f"{avg_dist:.1f}",
                'Clienti A+B aggiuntivi': '—',
                'Sat. Futura': '—'
            })

        editor_key = f"recv_editor_{removed_name}"
        edited = st.data_editor(
            pd.DataFrame(recv_data),
            column_config={
                "Seleziona": st.column_config.CheckboxColumn("Seleziona", default=False),
                "Ricevente": st.column_config.TextColumn("Ricevente", disabled=True),
                "Clienti A+B attuali": st.column_config.NumberColumn("Clienti A+B attuali", disabled=True),
                "Sat. Attuale": st.column_config.TextColumn("Sat. Attuale", disabled=True),
                "Distanza media (km) nuovi clienti": st.column_config.TextColumn("Distanza media (km)", disabled=True),
                "Clienti A+B aggiuntivi": st.column_config.TextColumn("Clienti A+B aggiuntivi", disabled=True),
                "Sat. Futura": st.column_config.TextColumn("Sat. Futura", disabled=True),
            },
            disabled=["Ricevente", "Clienti A+B attuali", "Sat. Attuale",
                     "Distanza media (km) nuovi clienti", "Clienti A+B aggiuntivi", "Sat. Futura"],
            hide_index=True,
            use_container_width=True,
            key=editor_key
        )

        selected_receivers = edited[edited['Seleziona']]['Ricevente'].tolist()

        calc_col1, calc_col2 = st.columns([1, 3])
        with calc_col1:
            calc_pressed = st.button(
                "🔄 Calcola Impatto",
                use_container_width=True,
                disabled=(not selected_receivers),
                key=f"btn_calc_{removed_name}"
            )

        if calc_pressed and selected_receivers:
            with st.spinner("Calcolo scenario in corso..."):
                abc = st.session_state.abc_vals
                df_temp = apply_reassignment(
                    st.session_state.df_c_working, removed_name, selected_receivers, df_v
                )
                temp_active = [r for r in reps if r not in st.session_state.removed_reps and rep_status.get(r, True)]

                df_temp_json = df_temp.to_json()
                df_v_json = df_v.to_json()
                temp_res, _ = _run_simulation_cached(
                    df_temp_json, df_v_json, temp_active, col_vol,
                    abc['threshold_a'],
                    abc['freq_a'], abc['freq_b'],
                    dur_visita, ore_effettive_gg, gg_lavoro,
                    max_stops_per_day,
                    pernotto_attivo, soglia_min_pernotto,
                    max_notti_week, max_notti_consecutive,
                    hotel_offset_km,
                    pausa_km, pausa_min_gg
                )

                alloc = preview_allocation(orphan_df, selected_receivers, st.session_state.df_c_working, df_v)

                future_results = {}
                for rep in selected_receivers:
                    fut_sat = None
                    if temp_res is not None and rep in temp_res['sales_rep'].values:
                        fut_sat = float(temp_res[temp_res['sales_rep'] == rep]['saturazione_pct'].iloc[0])
                    add_abc = 0
                    if rep in alloc and len(alloc[rep]) > 0:
                        assigned_indices = [i for i, _ in alloc[rep]]
                        sub_assigned = orphan_df.loc[assigned_indices]
                        add_abc = len(sub_assigned)
                    future_results[rep] = {
                        'fut_sat': fut_sat,
                        'add_abc': add_abc
                    }

                st.session_state[f"future_results_{removed_name}"] = future_results
                st.session_state[f"last_selected_{removed_name}"] = sorted(selected_receivers)
                st.rerun()

        future_results = st.session_state.get(f"future_results_{removed_name}", {})
        last_selected = st.session_state.get(f"last_selected_{removed_name}", [])

        if future_results and sorted(selected_receivers) == last_selected:
            result_rows = []
            for _, row in edited.iterrows():
                rep = row['Ricevente']
                if row['Seleziona'] and rep in future_results:
                    row_dict = row.to_dict()
                    row_dict['Clienti A+B aggiuntivi'] = future_results[rep]['add_abc']
                    row_dict['Sat. Futura'] = sat_emoji(future_results[rep]['fut_sat'])
                    result_rows.append(row_dict)
                else:
                    result_rows.append(row.to_dict())

            if result_rows:
                st.markdown("#### 📊 Impatto Simulato")
                result_df = pd.DataFrame(result_rows)
                result_df = result_df[['Seleziona', 'Ricevente', 'Clienti A+B attuali',
                                      'Sat. Attuale', 'Distanza media (km) nuovi clienti',
                                      'Clienti A+B aggiuntivi', 'Sat. Futura']]
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.caption("La saturazione futura è calcolata riassegnando gli orfani ai soli venditori selezionati.")
        elif future_results and sorted(selected_receivers) != last_selected:
            st.warning("⚠️ La selezione è cambiata. Clicca '🔄 Calcola Impatto' per aggiornare.")

        st.divider()
        st.markdown("### 📊 Preview allocazione clienti")

        if not selected_receivers:
            st.warning("Seleziona almeno un ricevente per vedere la preview")
        else:
            alloc = preview_allocation(orphan_df, selected_receivers, st.session_state.df_c_working, df_v)
            preview_headers = ['Ricevente', 'Clienti A', 'Clienti B', 'Totale Clienti', 'Volume Totale']
            preview_rows = []
            for r in selected_receivers:
                n_ass = len(alloc.get(r, []))
                if n_ass > 0:
                    assigned_indices = [idx for idx, _ in alloc[r]]
                    sub_assigned = orphan_df.loc[assigned_indices]
                    na = int((sub_assigned['classe'] == 'A').sum())
                    nb = int((sub_assigned['classe'] == 'B').sum())
                    vol_tot = sub_assigned[col_vol].sum()
                    preview_rows.append([
                        f"**{r}**",
                        str(na), str(nb),
                        str(n_ass),
                        fmt_eu(vol_tot)
                    ])

            if preview_rows:
                st.markdown(render_html_table(preview_headers, preview_rows, "12px"), unsafe_allow_html=True)
            else:
                st.info("Nessun cliente assegnato ai riceventi selezionati")

        st.divider()
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("✅ Conferma Riassegnazione", use_container_width=True, disabled=(not selected_receivers)):
                if selected_receivers:
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
                    st.session_state.pop(f"future_results_{removed_name}", None)
                    st.session_state.pop(f"last_selected_{removed_name}", None)
                    st.session_state.pending_removal = None
                    st.session_state.force_recalc = True
                    st.session_state.rep_status_prev = {r: rep_status[r] for r in reps}
                    st.rerun()
        with col_btn2:
            if st.button("❌ Annulla Rimozione", use_container_width=True):
                st.session_state[f"reset_rep_{removed_name}"] = True
                st.session_state.pending_removal = None
                st.rerun()
        st.stop()

    # =============================================================================
    # MAIN CONTENT: Matrice Classificazione (2 CLASSI)
    # =============================================================================
    if uploaded:
        st.divider()
        st.subheader("📊 Matrice Classificazione & Frequenze")
        st.caption("Definisci la soglia e le visite annue per le 2 classi di cliente.")

        # FIX: usa sempre i valori da session_state, mai None
        abc_defaults = st.session_state.abc_vals

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🟢 Class A**")
            threshold_a = st.number_input(
                "Soglia ≥", 
                key="threshold_a_input", 
                value=int(abc_defaults.get('threshold_a', 700)), 
                min_value=0, step=1
            )
            # FIX: freq_a visibile e modificabile
            freq_a = st.number_input(
                "Visite/anno Class A", 
                key="freq_a_input", 
                value=int(abc_defaults.get('freq_a', 20)), 
                min_value=0, step=1
            )
        with col2:
            st.markdown("**🔴 Class B**")
            st.number_input("Soglia ≥", value=0, disabled=True, key="threshold_b_disabled")
            st.number_input("Soglia <", value=int(threshold_a), disabled=True, key="threshold_b_max_disabled")
            freq_b = st.number_input(
                "Visite/anno Class B", 
                key="freq_b_input", 
                value=int(abc_defaults.get('freq_b', 12)), 
                min_value=0, step=1
            )

        st.session_state.abc_vals = {
            'threshold_a': threshold_a,
            'freq_a': freq_a,
            'freq_b': freq_b,
        }
        st.session_state.min_vol = 0

        # Preview distribuzione
        try:
            df_preview = df_c.copy()
            df_preview[col_vol] = pd.to_numeric(df_preview[col_vol], errors='coerce').fillna(0)
            df_preview = classify_abc(df_preview, col_vol, threshold_a)
            dist = df_preview['classe'].value_counts()
            col_prev1, col_prev2 = st.columns(2)
            with col_prev1: st.metric("🟢 Class A", f"{fmt_eu(dist.get('A', 0))}")
            with col_prev2: st.metric("🔴 Class B", f"{fmt_eu(dist.get('B', 0))}")
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
                    # FIX: assicurati che abc non sia None
                    if abc is None:
                        abc = {'threshold_a': 700, 'freq_a': 20, 'freq_b': 12}
                        st.session_state.abc_vals = abc

                    df_c_json = df_c.to_json()
                    df_v_json = df_v.to_json()
                    res, df_w = _run_simulation_cached(
                        df_c_json, df_v_json, active_list, col_vol,
                        abc['threshold_a'],
                        abc['freq_a'], abc['freq_b'],
                        dur_visita, ore_effettive_gg, gg_lavoro,
                        max_stops_per_day,
                        pernotto_attivo, soglia_min_pernotto,
                        max_notti_week, max_notti_consecutive,
                        hotel_offset_km,
                        pausa_km, pausa_min_gg
                    )

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

        st.info(f"📍 Stop/Giorno: **{params['max_stops']}** | Soglia Class A: **≥{fmt_eu(st.session_state.abc_vals['threshold_a'])}**")
        st.divider()

        st.subheader("📋 Dettaglio Scenario Corrente")
        with st.expander("ℹ️ Cosa significano le colonne?", expanded=False):
            st.markdown("""
            - **Time Capacity %** → Saturazione annua = Ore Totali / (Ore/Giorno × Giorni Lavoro)
            - **Min/Day** → Minuti medi di guida al giorno = Ore Viaggio Annue × 60 / Giorni Lavoro
            - **Visit/Day** → Visite medie al giorno = Ore Visite Annue × 60 / Durata Visita / Giorni Lavoro
            """)

        # Colonne richieste in inglese - 2 CLASSI
        disp = res[['sales_rep', 'stato', 'n_clienti', 'n_classe_a', 'n_classe_b',
                   'volume_a', 'volume_b', 'ore_visite_annue', 'ore_viaggio_annue', 'ore_totali_annue',
                   'saturazione_pct', 'driving_min_giorno', 'visite_giorno']].copy()
        disp.columns = ['Sales Representative', 'Status', 'Customers', 'Class A', 'Class B', 
                       'Volumes A', 'Volumes B', 'Visits Time', 'Driving Time', 'Total Time', 
                       'Time Capacity %', 'Min/Day', 'Visit/Day']

        # Riga TOTALE
        total_row = pd.DataFrame([{
            'Sales Representative': 'TOTAL',
            'Status': get_alert(disp['Time Capacity %'].mean()),
            'Customers': disp['Customers'].sum(),
            'Class A': disp['Class A'].sum(),
            'Class B': disp['Class B'].sum(),
            'Volumes A': disp['Volumes A'].sum(),
            'Volumes B': disp['Volumes B'].sum(),
            'Visits Time': disp['Visits Time'].sum(),
            'Driving Time': disp['Driving Time'].sum(),
            'Total Time': disp['Total Time'].sum(),
            'Time Capacity %': disp['Time Capacity %'].mean(),
            'Min/Day': disp['Min/Day'].mean(),
            'Visit/Day': disp['Visit/Day'].mean(),
        }])
        disp = pd.concat([disp, total_row], ignore_index=True)

        # Formattazione valori
        disp_fmt = disp.copy()
        for col in ['Customers', 'Class A', 'Class B', 'Volumes A', 'Volumes B']:
            disp_fmt[col] = disp_fmt[col].apply(lambda x: fmt_eu(x, 0))
        for col in ['Visits Time', 'Driving Time', 'Total Time', 'Min/Day']:
            disp_fmt[col] = disp_fmt[col].apply(lambda x: fmt_eu(x, 1))
        disp_fmt['Time Capacity %'] = disp_fmt['Time Capacity %'].apply(lambda x: fmt_eu(x, 1, '%'))
        disp_fmt['Visit/Day'] = disp_fmt['Visit/Day'].apply(lambda x: fmt_eu(x, 2))

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
                if col == 'Time Capacity %':
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
            st.caption(f"Visualizzati {fmt_eu(len(df_map))} clienti.")
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

            classe_colors = {'A': '#ff0000', 'B': '#0088ff'}
            for classe, color, size in [('A', classe_colors['A'], 6), ('B', classe_colors['B'], 4)]:
                df_cl = df_map[df_map['classe'] == classe]
                if len(df_cl) > 0:
                    fig.add_trace(go.Scattermapbox(
                        lat=df_cl['latitudine'], lon=df_cl['longitudine'],
                        mode='markers', marker=dict(size=size, color=color, opacity=0.8),
                        name=f"Class {classe}",
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

        # =============================================================================
        # EXPORT AVANZATO
        # =============================================================================
        st.divider()
        st.subheader("📦 Export Avanzato & Reportistica")

        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            if st.session_state.initial_fig is not None:
                if st.button("📸 Scarica Mappa Iniziale PNG", use_container_width=True):
                    img = fig_to_png(st.session_state.initial_fig)
                    if img:
                        st.download_button("⬇️ Download PNG", img, "mappa_status_quo_iniziale.png", "image/png", use_container_width=True)

        with col_ex2:
            if st.session_state.removal_figures:
                if st.button("📸 Scarica Mappe Riassegnazioni PNG", use_container_width=True):
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for rep_name, fig in st.session_state.removal_figures.items():
                            img = fig_to_png(fig)
                            if img:
                                zf.writestr(f"riassegnazione_{rep_name}.png", img)
                    zip_buffer.seek(0)
                    st.download_button("⬇️ Download ZIP Mappe", zip_buffer.getvalue(), "mappe_riassegnazioni.zip", "application/zip", use_container_width=True)

        with col_ex3:
            if st.button("🏁 Status Quo Finale + Export Excel", use_container_width=True, type="primary"):
                with st.spinner("Generazione report finale..."):
                    final_fig = build_territory_map(
                        st.session_state.current_df_work,
                        st.session_state.current_df_v,
                        st.session_state.current_params['active_list'],
                        title="🗺️ Status Quo Finale"
                    )
                    st.session_state.final_fig = final_fig

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
                        img_final = fig_to_png(final_fig)
                        if img_final:
                            st.download_button("📥 Scarica Mappa Finale PNG", img_final, "mappa_status_quo_finale.png", "image/png", use_container_width=True)
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
    # SALVATAGGIO STATO
    # =============================================================================
    if uploaded and not st.session_state.get('pending_removal'):
        st.session_state.rep_status_prev = {r: rep_status[r] for r in reps}

if __name__ == "__main__":
    main()
