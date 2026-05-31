import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🎯 Field Force Downsizing Simulator", layout="wide", page_icon="📊")

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

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

def classify_abc(df, volume_col, thr_a, thr_b):
    df = df.copy()
    conditions = [
        df[volume_col] >= thr_a,
        (df[volume_col] >= thr_b) & (df[volume_col] < thr_a),
        df[volume_col] < thr_b
    ]
    choices = ['A', 'B', 'C']
    df['classe'] = np.select(conditions, choices, default='C')
    return df

def compute_convex_hull(df, lat_col, lon_col):
    if len(df) < 3:
        return None, None
    try:
        points = df[[lon_col, lat_col]].values
        hull = ConvexHull(points)
        hull_idx = np.append(hull.vertices, hull.vertices[0])
        return points[hull_idx, 0], points[hull_idx, 1]
    except:
        return None, None

def main():
    st.title("🎯 Field Force Downsizing Simulator")
    st.markdown("*Strumento strategico per ottimizzazione rete vendita Italia*")
    
    with st.sidebar:
        st.subheader("📁 Dati di Input")
        uploaded_file = st.file_uploader("Carica Excel", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👆 Carica il file Excel")
            return
        
        try:
            df_c = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
            df_v = pd.read_excel(uploaded_file, sheet_name="venditori")
        except Exception as e:
            st.error(f"❌ Errore lettura: {str(e)}")
            return
        
        df_c.columns = [c.strip().lower() for c in df_c.columns]
        df_v.columns = [c.strip().lower() for c in df_v.columns]
        
        required_cust = ['latitudine', 'longitudine', 'sigla']
        required_rep = ['latitudine', 'longitudine', 'sales rep']
        
        if not all(c in df_c.columns for c in required_cust):
            st.error(f"❌ Clienti: mancano {set(required_cust) - set(df_c.columns)}")
            return
        if not all(c in df_v.columns for c in required_rep):
            st.error(f"❌ Venditori: mancano {set(required_rep) - set(df_v.columns)}")
            return
        
        for col in ['latitudine', 'longitudine']:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            df_v[col] = pd.to_numeric(df_v[col], errors='coerce')
        df_c = df_c.dropna(subset=['latitudine', 'longitudine'])
        
        st.success(f"✅ Caricati: {len(df_c):,} clienti, {len(df_v):,} venditori")
        st.divider()
        
        st.subheader("🎛️ Parametri")
        
        volume_cols = [c for c in df_c.columns if any(k in c.lower() for k in ['gy', 'du', 'tot', '25', '26'])]
        col_volume = st.selectbox("Colonna Volume", volume_cols if volume_cols else df_c.columns.tolist())
        
        min_volume = st.number_input("Taglia minima cliente", 0, 100000, 0, step=100)
        thr_a = st.number_input("Soglia A", 1000, 10000, 3000, step=500)
        thr_b = st.number_input("Soglia B", 200, 2999, 800, step=100)
        
        freq_a = st.number_input("Visite/anno - A", 6, 36, 12, step=2)
        freq_b = st.number_input("Visite/anno - B", 3, 18, 6, step=1)
        freq_c = st.number_input("Visite/anno - C", 1, 6, 3, step=1)
        durata_visita_min = st.slider("Durata visita (min)", 40, 150, 90, step=5)
        
        ore_giorno = st.number_input("Ore/giorno", 6.0, 10.0, 8.0, step=0.5)
        giorni_anno = st.number_input("Giorni/anno", 200, 260, 220, step=5)
        max_ore_annue = ore_giorno * giorni_anno
        
        vel_media = st.slider("Velocità media km/h", 40, 100, 65, step=5)
        
        st.subheader("👥 Venditori Attivi")
        v_list = sorted(df_v['sales rep'].unique())
        active_reps = {v: st.checkbox(v, value=True) for v in v_list}
        
        run_btn = st.button("🚀 CALCOLA", type="primary", use_container_width=True)
    
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'df_work' not in st.session_state:
        st.session_state.df_work = None
    
    if uploaded_file and (run_btn or st.session_state.result_df is None):
        with st.spinner("Calcolo in corso..."):
            try:
                active_list = [v for v, s in active_reps.items() if s]
                if not active_list:
                    st.error("Seleziona almeno 1 venditore")
                    return
                
                df_active_v = df_v[df_v['sales rep'].isin(active_list)].copy()
                df_work = df_c[df_c[col_volume] >= min_volume].copy()
                
                if len(df_work) == 0:
                    st.error(f"Nessun cliente >= {min_volume:,}")
                    return
                
                df_work = classify_abc(df_work, col_volume, thr_a, thr_b)
                df_work['tortuosity'] = df_work['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)
                
                # Assegnazione nearest neighbor
                c_coords = df_work[['longitudine', 'latitudine']].values
                v_coords = df_active_v[['longitudine', 'latitudine']].values
                
                dist_matrix = haversine_km(
                    c_coords[:, 0][:, None], c_coords[:, 1][:, None],
                    v_coords[:, 0][None, :], v_coords[:, 1][None, :]
                )
                
                idx_min = np.argmin(dist_matrix, axis=1)
                df_work['assigned_rep'] = df_active_v.iloc[idx_min]['sales rep'].values
                df_work['dist_km_oneway'] = dist_matrix[np.arange(len(df_work)), idx_min]
                df_work['dist_km_round'] = df_work['dist_km_oneway'] * df_work['tortuosity'] * 2
                
                # Frequenze
                freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c}
                df_work['freq_visite'] = df_work['classe'].map(freq_map)
                
                # CALCOLO CORRETTO: ore per cliente = (freq × durata) / 60
                df_work['ore_visita_per_anno'] = (df_work['freq_visite'] * durata_visita_min) / 60.0
                df_work['ore_viaggio_per_anno'] = df_work['dist_km_round'] / vel_media
                df_work['ore_totali_per_anno'] = df_work['ore_visita_per_anno'] + df_work['ore_viaggio_per_anno']
                
                # Aggregazione
                agg = df_work.groupby('assigned_rep').agg(
                    n_clienti=('assigned_rep', 'count'),
                    n_classe_a=('classe', lambda x: (x=='A').sum()),
                    n_classe_b=('classe', lambda x: (x=='B').sum()),
                    n_classe_c=('classe', lambda x: (x=='C').sum()),
                    volume_totale=(col_volume, 'sum'),
                    ore_visite_annue=('ore_visita_per_anno', 'sum'),
                    ore_viaggio_annue=('ore_viaggio_per_anno', 'sum'),
                    ore_totali_annue=('ore_totali_per_anno', 'sum'),
                    dist_media=('dist_km_oneway', 'mean')
                ).reset_index().rename(columns={'assigned_rep': 'sales_rep'})
                
                agg['saturazione_pct'] = (agg['ore_totali_annue'] / max_ore_annue) * 100
                agg['driving_min_giorno'] = (agg['ore_viaggio_annue'] * 60) / giorni_anno
                agg['visite_giorno'] = (agg['ore_visite_annue'] * 60 / durata_visita_min) / giorni_anno
                
                def get_alert(sat):
                    if sat > 110: return "🔴 CRITICO"
                    elif sat > 100: return "🟠 OVERLOAD"
                    elif sat > 85: return "🟡 ATTENZIONE"
                    else: return "🟢 OK"
                agg['stato'] = agg['saturazione_pct'].apply(get_alert)
                
                all_reps = pd.DataFrame({'sales_rep': active_list})
                result = all_reps.merge(agg, on='sales_rep', how='left').fillna(0)
                result = result.sort_values('saturazione_pct', ascending=False)
                
                st.session_state.result_df = result
                st.session_state.df_work = df_work
                st.session_state.df_active_v = df_active_v
                st.session_state.active_list = active_list
                st.session_state.max_ore = max_ore_annue
                st.session_state.giorni_anno = giorni_anno
                
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")
                st.exception(e)
                return
    
    if st.session_state.result_df is not None:
        result = st.session_state.result_df
        df_work = st.session_state.df_work
        active_list = st.session_state.active_list
        
        st.subheader("📊 KPI Scenario")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Venditori Attivi", len(active_list))
        c2.metric("Clienti Serviti", f"{int(result['n_clienti'].sum()):,}")
        c3.metric("Saturazione Media", f"{result['saturazione_pct'].mean():.1f}%")
        c4.metric("Overload", len(result[result['saturazione_pct']>100]), delta_color="inverse")
        
        st.subheader("📋 Dettaglio Venditori")
        disp_cols = {
            'sales_rep': 'Venditore', 'stato': 'Stato', 'n_clienti': 'Clienti',
            'n_classe_a': 'A', 'n_classe_b': 'B', 'n_classe_c': 'C',
            'volume_totale': 'Volume', 'ore_visite_annue': 'Ore Visite',
            'ore_viaggio_annue': 'Ore Viaggio', 'ore_totali_annue': 'Ore Totali',
            'saturazione_pct': 'Saturazione %', 'driving_min_giorno': 'Min Guida/Giorno',
            'visite_giorno': 'Visite/Giorno'
        }
        
        df_disp = result[list(disp_cols.keys())].copy()
        df_disp.columns = [disp_cols[c] for c in df_disp.columns]
        
        def color_sat(val):
            if pd.isna(val): return ''
            try:
                n = float(str(val).replace('%',''))
                if n > 110: return 'background-color:#ffcdd2;color:#b71c1c'
                elif n > 100: return 'background-color:#ffe0b2;color:#e65100'
                elif n > 85: return 'background-color:#fff9c4;color:#f57f17'
                else: return 'background-color:#c8e6c9;color:#1b5e20'
            except: return ''
        
        styled = df_disp.style.map(color_sat, subset=['Saturazione %']).format({
            'Volume':'{:,.0f}', 'Ore Visite':'{:.1f}', 'Ore Viaggio':'{:.1f}',
            'Ore Totali':'{:.1f}', 'Saturazione %':'{:.1f}%', 'Min Guida/Giorno':'{:.1f}',
            'Visite/Giorno':'{:.2f}'
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
        st.subheader("🗺️ Mappa Territori")
        show_hull = st.checkbox("Mostra confini", value=True)
        
        fig = px.scatter_mapbox(df_work, lat="latitudine", lon="longitudine",
                                color="assigned_rep", size="freq_visite", size_max=8,
                                hover_data={"classe":True, "freq_visite":True, 
                                           "latitudine":False, "longitudine":False},
                                zoom=5, height=600, render_mode="webgl")
        
        if show_hull:
            colors = px.colors.qualitative.Set3
            for i, rep in enumerate(active_list):
                subset = df_work[df_work['assigned_rep']==rep]
                lon_h, lat_h = compute_convex_hull(subset, 'latitudine', 'longitudine')
                if lon_h is not None:
                    fig.add_trace(go.Scattermapbox(mode="lines", lon=lon_h, lat=lat_h,
                                   line=dict(width=2, color=colors[i%len(colors)]),
                                   name=rep, showlegend=True))
        
        fig.add_trace(go.Scattermapbox(lat=st.session_state.df_active_v['latitudine'],
                       lon=st.session_state.df_active_v['longitudine'], mode='markers+text',
                       marker=dict(size=14, symbol='star', color='black'),
                       text=st.session_state.df_active_v['sales rep'],
                       textposition="top center", name='🏠 Home', hoverinfo='text'))
        
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0},
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔍 Dettaglio Operativo")
        sel_rep = st.selectbox("Seleziona venditore", active_list)
        if sel_rep:
            rd = result[result['sales_rep']==sel_rep].iloc[0]
            rc = df_work[df_work['assigned_rep']==sel_rep]
            
            cd1, cd2 = st.columns([1,2])
            with cd1:
                st.markdown(f"**{sel_rep}**")
                st.markdown(f"- Stato: {rd['stato']}")
                st.markdown(f"- Clienti: {int(rd['n_clienti'])} (A:{int(rd['n_classe_a'])} B:{int(rd['n_classe_b'])} C:{int(rd['n_classe_c'])})")
                st.markdown(f"- Volume: {rd['volume_totale']:,.0f}")
                st.markdown(f"- Ore Visite/anno: {rd['ore_visite_annue']:.1f}")
                st.markdown(f"- Ore Viaggio/anno: {rd['ore_viaggio_annue']:.1f}")
                st.markdown(f"- Saturazione: {rd['saturazione_pct']:.1f}%")
                st.markdown(f"- Guida/giorno: {rd['driving_min_giorno']:.1f} min")
                if rd['driving_min_giorno'] > 180:
                    st.warning(f"⚠️ Guida elevata!")
            
            with cd2:
                if len(rc) > 0:
                    fz = px.scatter_mapbox(rc, lat="latitudine", lon="longitudine",
                           color="classe", color_discrete_map={'A':'red','B':'orange','C':'green'},
                           size="freq_visite", zoom=7, height=300, render_mode="webgl")
                    rh = st.session_state.df_active_v[st.session_state.df_active_v['sales rep']==sel_rep]
                    if len(rh)>0:
                        fz.add_trace(go.Scattermapbox(lat=rh['latitudine'], lon=rh['longitudine'],
                                      mode='markers', marker=dict(size=15, symbol='star', color='black')))
                    fz.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fz, use_container_width=True)
        
        st.divider()
        csv = result.to_csv(index=False, decimal=';', sep=';')
        st.download_button("📥 Scarica CSV", data=csv,
                          file_name=f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                          mime="text/csv", use_container_width=True)
    else:
        st.info("👈 Clicca CALCOLA per avviare")

if __name__ == "__main__":
    main()
