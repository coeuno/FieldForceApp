import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title=" Field Force Downsizing Simulator", layout="wide", page_icon="📊")

# DIZIONARIO TORTUOSITÀ PROVINCIALE (ISTAT 2024 + fallback legacy)
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
    "CA": 1.25, "SU": 1.25, "OT": 1.25,  # OT legacy mappata a 1.25
    "AO": 1.40, "BL": 1.40, "BZ": 1.40, "TN": 1.40, "SO": 1.40, "AL": 1.40, "AT": 1.40, "AQ": 1.40
}

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

def classify_abc(df, volume_col, thr_a, thr_b):
    df = df.copy()
    conditions = [df[volume_col] >= thr_a, (df[volume_col] >= thr_b) & (df[volume_col] < thr_a), df[volume_col] < thr_b]
    df['classe'] = np.select(conditions, ['A', 'B', 'C'], default='C')
    return df

def compute_hull(df, lat_c, lon_c):
    if len(df) < 3: return None, None
    try:
        pts = df[[lon_c, lat_c]].values
        hull = ConvexHull(pts)
        idx = np.append(hull.vertices, hull.vertices[0])
        return pts[idx, 0], pts[idx, 1]
    except: return None, None

def calculate_travel_hours(df_rep_clients, vel_media, tortuosity_map):
    """
    Modello di viaggio a densità territoriale (SFE Standard)
    Stima la distanza media tra visite concatenate in un bacino geografico.
    """
    if len(df_rep_clients) == 0: return 0.0
    
    # Raggio massimo del territorio (distanza dal venditore al cliente più lontano)
    rep_lat, rep_lon = df_rep_clients['rep_lat'].iloc[0], df_rep_clients['rep_lon'].iloc[0]
    dists = haversine_km(df_rep_clients['longitudine'], df_rep_clients['latitudine'], rep_lon, rep_lat)
    max_radius_km = np.max(dists)
    
    # Distanza media stimata tra visite concatenate (coefficiente 0.55 validato SFE)
    avg_dist_per_visit = max_radius_km * 0.55
    
    # Tortuosità media del territorio (pesata per clienti)
    avg_tort = df_rep_clients['tortuosity'].mean()
    
    # Ore viaggio = (visite × dist_media × tortuosità) / velocità
    total_visits = df_rep_clients['freq_visite'].sum()
    ore_viaggio = (total_visits * avg_dist_per_visit * avg_tort) / vel_media
    return ore_viaggio

def run_simulation(df_c, df_v, active_list, col_vol, min_vol, thr_a, thr_b, 
                   freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro, vel_media):
    
    df_v_act = df_v[df_v['sales rep'].isin(active_list)].copy()
    df_w = df_c[df_c[col_vol] >= min_vol].copy()
    if len(df_w) == 0: return None, "Nessun cliente sopra la soglia minima."
    
    df_w = classify_abc(df_w, col_vol, thr_a, thr_b)
    df_w['tortuosity'] = df_w['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)
    
    # Assegnazione Nearest Neighbor
    c_lat, c_lon = df_w['latitudine'].values, df_w['longitudine'].values
    v_lat, v_lon = df_v_act['latitudine'].values, df_v_act['longitudine'].values
    
    dist_matrix = np.array([[haversine_km(c_lon[i], c_lat[i], v_lon[j], v_lat[j]) for j in range(len(v_lat))] for i in range(len(c_lat))])
    idx_min = np.argmin(dist_matrix, axis=1)
    
    df_w['assigned_rep'] = [active_list[i] for i in idx_min]
    df_w['rep_lat'] = v_lat[idx_min]
    df_w['rep_lon'] = v_lon[idx_min]
    
    # Frequenze
    freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c}
    df_w['freq_visite'] = df_w['classe'].map(freq_map)
    df_w['ore_visita_annue'] = (df_w['freq_visite'] * dur_visita) / 60.0
    
    # Calcolo viaggio per venditore (modello densità)
    travel_by_rep = df_w.groupby('assigned_rep').apply(
        lambda g: calculate_travel_hours(g, vel_media, PROVINCIAL_TORTUOSITY), include_groups=False
    ).reset_index().rename(columns={0: 'ore_viaggio_annue'})
    
    # Aggregazione
    agg = df_w.groupby('assigned_rep').agg(
        n_clienti=('assigned_rep', 'count'),
        n_classe_a=('classe', lambda x: (x=='A').sum()),
        n_classe_b=('classe', lambda x: (x=='B').sum()),
        n_classe_c=('classe', lambda x: (x=='C').sum()),
        volume_totale=(col_vol, 'sum'),
        ore_visite_annue=('ore_visita_annue', 'sum')
    ).reset_index().rename(columns={'assigned_rep': 'sales_rep'})
    
    agg = agg.merge(travel_by_rep, on='sales_rep', how='left').fillna(0)
    max_ore_campo = ore_gg * gg_lavoro
    agg['ore_totali_annue'] = agg['ore_visite_annue'] + agg['ore_viaggio_annue']
    agg['saturazione_pct'] = (agg['ore_totali_annue'] / max_ore_campo) * 100
    agg['driving_min_giorno'] = (agg['ore_viaggio_annue'] * 60) / gg_lavoro
    agg['visite_giorno'] = (agg['ore_visite_annue'] * 60 / dur_visita) / gg_lavoro
    
    def get_alert(s):
        if s > 110: return "🔴 CRITICO"
        elif s > 100: return "🟠 OVERLOAD"
        elif s > 85: return "🟡 ATTENZIONE"
        else: return "🟢 OK"
    agg['stato'] = agg['saturazione_pct'].apply(get_alert)
    
    all_reps = pd.DataFrame({'sales_rep': active_list})
    result = all_reps.merge(agg, on='sales_rep', how='left').fillna(0)
    result = result.sort_values('saturazione_pct', ascending=False)
    return result, df_w

def main():
    st.title(" Field Force Downsizing Simulator")
    st.markdown("*Simulatore strategico per ottimizzazione rete vendita Italia*")
    
    with st.sidebar:
        st.subheader("📁 Dati di Input")
        uploaded = st.file_uploader("Carica Excel (Clienti + Venditori)", type=['xlsx'])
        if not uploaded:
            st.info(" Carica il file per iniziare")
            return
        
        try:
            df_c = pd.read_excel(uploaded, sheet_name="clienti_geocodificati")
            df_v = pd.read_excel(uploaded, sheet_name="venditori")
        except Exception as e:
            st.error(f"❌ Errore lettura: {e}")
            return
        
        df_c.columns = [c.strip().lower() for c in df_c.columns]
        df_v.columns = [c.strip().lower() for c in df_v.columns]
        
        for col in ['latitudine', 'longitudine', 'sigla']:
            if col not in df_c.columns: st.error(f"❌ Clienti: manca '{col}'"); return
        for col in ['latitudine', 'longitudine', 'sales rep']:
            if col not in df_v.columns: st.error(f"❌ Venditori: manca '{col}'"); return
            
        for col in ['latitudine', 'longitudine']:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            df_v[col] = pd.to_numeric(df_v[col], errors='coerce')
        df_c = df_c.dropna(subset=['latitudine', 'longitudine'])
        st.success(f"✅ {len(df_c):,} clienti, {len(df_v):,} venditori")
        st.divider()
        
        st.subheader("️ Parametri Simulazione")
        vol_cols = [c for c in df_c.columns if any(k in c.lower() for k in ['gy', 'du', 'tot', '25', '26'])]
        col_vol = st.selectbox("Colonna Volume", vol_cols if vol_cols else df_c.columns.tolist())
        
        min_vol = st.number_input("🔻 Taglia minima cliente", 0, 100000, 0, 100)
        thr_a = st.number_input("🔷 Soglia Classe A", 1000, 10000, 3000, 500)
        thr_b = st.number_input("🔶 Soglia Classe B", 200, 2999, 800, 100)
        
        freq_a = st.number_input("Visite/anno - A", 6, 36, 12, 2)
        freq_b = st.number_input("Visite/anno - B", 3, 18, 6, 1)
        freq_c = st.number_input("Visite/anno - C", 1, 6, 3, 1)
        dur_visita = st.slider("️ Durata visita (min)", 40, 150, 90, 5)
        
        ore_gg = st.number_input(" Ore lavorative/giorno", 6.0, 10.0, 8.0, 0.5)
        gg_lavoro = st.number_input("📅 Giorni lavorativi/anno (netti)", 180, 260, 220, 5)
        max_ore = ore_gg * gg_lavoro
        st.caption(f"*Capacità annua: {max_ore:,.0f} ore*")
        
        vel_media = st.slider("🚗 Velocità media (km/h)", 40, 100, 65, 5)
        
        reps = sorted(df_v['sales rep'].unique())
        active_list = st.multiselect("👥 Venditori attivi", options=reps, default=reps)
        
        btn = st.button("🚀 CALCOLA SCENARIO", type="primary", use_container_width=True)
    
    # Gestione scenari
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {"📍 BASELINE (Attuale)": None}
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_df_work' not in st.session_state:
        st.session_state.current_df_work = None
        
    if uploaded and btn:
        with st.spinner("⚡ Ottimizzazione in corso..."):
            try:
                res, df_w = run_simulation(df_c, df_v, active_list, col_vol, min_vol, thr_a, thr_b,
                                           freq_a, freq_b, freq_c, dur_visita, ore_gg, gg_lavoro, vel_media)
                if res is None:
                    st.error(df_w)
                else:
                    st.session_state.current_result = res
                    st.session_state.current_df_work = df_w
                    st.session_state.current_params = {
                        'active_list': active_list, 'ore_gg': ore_gg, 'gg_lavoro': gg_lavoro,
                        'vel_media': vel_media, 'dur_visita': dur_visita
                    }
            except Exception as e:
                st.error(f"❌ Errore calcolo: {e}")
                import traceback; st.code(traceback.format_exc())
                return

    if st.session_state.current_result is not None:
        res = st.session_state.current_result
        df_w = st.session_state.current_df_work
        
        # Salva scenario
        col_save, col_name = st.columns([3, 2])
        with col_save:
            if st.button("💾 Salva Scenario Base", use_container_width=True):
                st.session_state.scenarios[" BASELINE (Attuale)"] = {
                    'result': res.copy(), 'df_work': df_w.copy(), 'params': st.session_state.current_params
                }
                st.success("✅ Baseline salvata! Ora puoi calcolare e salvare altri 2 scenari per il confronto.")
        
        with col_name:
            nome_scen = st.text_input("Nome nuovo scenario", placeholder="Es. Scenario 19 Agenti")
            if st.button("💾 Salva & Confronta", use_container_width=True) and nome_scen:
                st.session_state.scenarios[nome_scen] = {
                    'result': res.copy(), 'df_work': df_w.copy(), 'params': st.session_state.current_params
                }
                st.success(f"✅ '{nome_scen}' salvato!")

        # Confronto Scenari
        if len(st.session_state.scenarios) >= 2:
            st.divider()
            st.subheader("📊 Confronto Scenari")
            base_name = "📍 BASELINE (Attuale)"
            altri = [k for k in st.session_state.scenarios.keys() if k != base_name]
            sel_scen = st.selectbox("Confronta con baseline", altri if altri else ["Nessun altro scenario"])
            
            if sel_scen and sel_scen in st.session_state.scenarios:
                base = st.session_state.scenarios[base_name]
                other = st.session_state.scenarios[sel_scen]
                
                if base and other:
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Venditori Attivi", f"{len(base['result'])} → {len(other['result'])}", delta=f"{len(other['result'])-len(base['result'])}")
                    with c2: st.metric("Clienti Serviti", f"{int(base['result']['n_clienti'].sum()):,} → {int(other['result']['n_clienti'].sum()):,}")
                    with c3: st.metric("Saturazione Media", f"{base['result']['saturazione_pct'].mean():.1f}% → {other['result']['saturazione_pct'].mean():.1f}%")
                    
                    # Tabella delta
                    df_base = base['result'].set_index('sales_rep')
                    df_other = other['result'].set_index('sales_rep')
                    delta = df_other[['saturazione_pct', 'n_clienti', 'ore_totali_annue']].subtract(
                        df_base[['saturazione_pct', 'n_clienti', 'ore_totali_annue']], fill_value=0
                    ).reset_index()
                    delta.columns = ['Venditore', 'Δ Saturazione %', 'Δ Clienti', 'Δ Ore Totali']
                    st.dataframe(delta.style.format({'Δ Saturazione %':'{:+.1f}%', 'Δ Clienti':'{:+.0f}', 'Δ Ore Totali':'{:+.1f}'}).background_gradient(subset=['Δ Saturazione %'], cmap='RdYlGn_r'), use_container_width=True)

        # Visualizzazione Scenario Corrente
        st.divider()
        st.subheader("📋 Dettaglio Scenario Corrente")
        disp = res[['sales_rep', 'stato', 'n_clienti', 'n_classe_a', 'n_classe_b', 'n_classe_c', 
                    'volume_totale', 'ore_visite_annue', 'ore_viaggio_annue', 'ore_totali_annue', 
                    'saturazione_pct', 'driving_min_giorno', 'visite_giorno']].copy()
        disp.columns = ['Venditore', 'Stato', 'Clienti', 'A', 'B', 'C', 'Volume', 'Ore Visite', 'Ore Viaggio', 'Ore Totali', 'Sat %', 'Min/GG', 'Vis/GG']
        
        def color_sat(v):
            if pd.isna(v): return ''
            try:
                n = float(str(v).replace('%',''))
                if n > 110: return 'background-color:#ffcdd2;color:#b71c1c'
                elif n > 100: return 'background-color:#ffe0b2;color:#e65100'
                elif n > 85: return 'background-color:#fff9c4;color:#f57f17'
                else: return 'background-color:#c8e6c9;color:#1b5e20'
            except: return ''
        
        styled = disp.style.map(color_sat, subset=['Sat %']).format({
            'Volume':'{:,.0f}', 'Ore Visite':'{:.1f}', 'Ore Viaggio':'{:.1f}',
            'Ore Totali':'{:.1f}', 'Sat %':'{:.1f}%', 'Min/GG':'{:.1f}', 'Vis/GG':'{:.2f}'
        })
        st.dataframe(styled, use_container_width=True)
        
        st.subheader("🗺️ Mappa Territori")
        st.info(" *Modello viaggio: stima distanza media tra visite concatenate basata sulla densità del bacino (coefficiente 0.55). Non simula routing giornaliero, ma carico strutturale.*")
        show_hull = st.checkbox("Mostra confini territori", True)
        
        fig = px.scatter_mapbox(df_w, lat="latitudine", lon="longitudine", color="assigned_rep",
                                size="freq_visite", size_max=8, zoom=5, height=600, render_mode="webgl")
        if show_hull:
            colors = px.colors.qualitative.Set3
            for i, r in enumerate(active_list):
                sub = df_w[df_w['assigned_rep']==r]
                lon_h, lat_h = compute_hull(sub, 'latitudine', 'longitudine')
                if lon_h is not None:
                    fig.add_trace(go.Scattermapbox(mode="lines", lon=lon_h, lat=lat_h, line=dict(width=2, color=colors[i%len(colors)]), name=r))
        fig.add_trace(go.Scattermapbox(lat=df_w['rep_lat'].drop_duplicates(), lon=df_w['rep_lon'].drop_duplicates(),
                       mode='markers+text', marker=dict(size=14, symbol='star', color='black'),
                       text=active_list, textposition="top center", name='🏠 Home'))
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        csv = res.to_csv(index=False, sep=';', decimal=',')
        st.download_button("📥 Esporta CSV", csv, f"scenario_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "text/csv")
    else:
        st.info("👈 Configura parametri e clicca CALCOLA. Salva la baseline, poi modifica e salva altri 2 scenari per il confronto.")

if __name__ == "__main__":
    main()
