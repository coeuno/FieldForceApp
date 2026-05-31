import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="🎯 Field Force Downsizing Simulator", layout="wide", page_icon="📊")

# --- DIZIONARIO TORTUOSITÀ PROVINCIALE ---
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

# --- FUNZIONI CORE ---
def haversine_km(lon1, lat1, lon2, lat2):
    """Calcolo vettoriale distanza geodetica in km"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))

def classify_abc(df, volume_col, thr_a, thr_b):
    """Classificazione ABC dinamica"""
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
    """Calcola poligono Convex Hull"""
    if len(df) < 3:
        return None, None
    try:
        points = df[[lon_col, lat_col]].values
        hull = ConvexHull(points)
        hull_idx = np.append(hull.vertices, hull.vertices[0])
        return points[hull_idx, 0], points[hull_idx, 1]
    except:
        return None, None

# --- INTERFACCIA PRINCIPALE ---
def main():
    st.title("🎯 Field Force Downsizing Simulator")
    st.markdown("*Strumento strategico per ottimizzazione rete vendita Italia*")
    
    # Sidebar
    with st.sidebar:
        st.subheader("📁 Dati di Input")
        uploaded_file = st.file_uploader("Carica Excel (Clienti + Venditori)", type=['xlsx', 'xls'])
        
        if not uploaded_file:
            st.info("👆 Carica il file per iniziare")
            return
        
        # Caricamento dati
        try:
            df_c = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
            df_v = pd.read_excel(uploaded_file, sheet_name="venditori")
        except Exception as e:
            st.error(f"❌ Errore lettura file: {str(e)}")
            return
        
        # Normalizzazione colonne
        df_c.columns = [c.strip().lower() for c in df_c.columns]
        df_v.columns = [c.strip().lower() for c in df_v.columns]
        
        # Validazione colonne
        required_cust = ['latitudine', 'longitudine', 'sigla']
        required_rep = ['latitudine', 'longitudine', 'sales rep']
        if not all(c in df_c.columns for c in required_cust):
            st.error(f"❌ Clienti: mancano colonne {set(required_cust) - set(df_c.columns)}")
            return
        if not all(c in df_v.columns for c in required_rep):
            st.error(f"❌ Venditori: mancano colonne {set(required_rep) - set(df_v.columns)}")
            return
        
        # Conversione coordinate
        for col in ['latitudine', 'longitudine']:
            df_c[col] = pd.to_numeric(df_c[col], errors='coerce')
            df_v[col] = pd.to_numeric(df_v[col], errors='coerce')
        df_c = df_c.dropna(subset=['latitudine', 'longitudine'])
        
        st.success(f"✅ Dati caricati: {len(df_c):,} clienti, {len(df_v):,} venditori")
        
        st.divider()
        st.subheader("🎛️ Parametri di Simulazione")
        
        # Parametri Portfolio
        st.markdown("**📦 Parametri Portfolio**")
        volume_cols = [c for c in df_c.columns if any(k in c.lower() for k in ['25', '26', 'tot', 'vol', 'pezzi'])]
        col_volume = st.selectbox("Metrica di Valutazione", volume_cols if volume_cols else df_c.columns.tolist(), index=0 if volume_cols else 0)
        
        min_volume = st.number_input("🔻 Taglia Minima Cliente", 0, 10000, 0, step=50)
        thr_a = st.number_input("🔷 Soglia Classe A (pezzi/anno)", 100, 5000, 500, step=50)
        thr_b = st.number_input("🔶 Soglia Classe B (pezzi/anno)", 50, 499, 200, step=10)
        
        # Parametri Carico
        st.markdown("**⏱️ Frequenza Visite**")
        freq_a = st.number_input("Visite/anno - Classe A", 6, 48, 12, step=2)
        freq_b = st.number_input("Visite/anno - Classe B", 2, 24, 6, step=1)
        freq_c = st.number_input("Visite/anno - Classe C", 1, 12, 3, step=1)
        durata_visita_min = st.slider("⏱️ Durata media visita (minuti)", 40, 150, 90, step=5)
        
        # Parametri Workload
        st.markdown("**👤 Capacità Venditore**")
        ore_giorno = st.number_input("Ore lavorative/giorno", 6.0, 10.0, 8.0, step=0.5)
        giorni_lavoro_anno = st.number_input("Giorni lavorativi/anno", 200, 260, 220, step=5)
        max_ore_annue = ore_giorno * giorni_lavoro_anno
        st.caption(f"*Capacità annua: {max_ore_annue:,.0f} ore*")
        
        # Parametri Logistici
        st.markdown("**🚗 Logistica**")
        vel_media = st.slider("Velocità media (km/h)", 40, 100, 65, step=5)
        
        # Selezione Venditori
        st.markdown("**👥 Configurazione Team**")
        v_list = sorted(df_v['sales rep'].unique())
        active_reps = {v: st.checkbox(v, value=True) for v in v_list}
        
        # Bottone calcolo
        run_btn = st.button("🚀 CALCOLA SCENARIO", type="primary", use_container_width=True)
    
    # Inizializza stato sessione
    if 'last_calc' not in st.session_state:
        st.session_state.last_calc = None
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'df_work' not in st.session_state:
        st.session_state.df_work = None
    
    # Calcolo automatico al caricamento o al click
    if uploaded_file and (run_btn or st.session_state.result_df is None):
        with st.spinner("⚡ Calcolo in corso..."):
            try:
                # Preparazione dati
                active_list = [v for v, status in active_reps.items() if status]
                if len(active_list) == 0:
                    st.error("⚠️ Seleziona almeno un venditore")
                    return
                
                df_active_v = df_v[df_v['sales rep'].isin(active_list)].copy()
                
                # Filtro clienti
                df_work = df_c[df_c[col_volume] >= min_volume].copy()
                if len(df_work) == 0:
                    st.error(f"⚠️ Nessun cliente sopra {min_volume:,}")
                    return
                
                # Classificazione ABC
                df_work = classify_abc(df_work, col_volume, thr_a, thr_b)
                
                # Tortuosità
                df_work['tortuosity'] = df_work['sigla'].map(PROVINCIAL_TORTUOSITY).fillna(1.25)
                
                # Assegnazione Nearest Neighbor
                c_coords = df_work[['longitudine', 'latitudine']].values
                v_coords = df_active_v[['longitudine', 'latitudine']].values
                
                dist_matrix = haversine_km(
                    c_coords[:, 0][:, None], c_coords[:, 1][:, None],
                    v_coords[:, 0][None, :], v_coords[:, 1][None, :]
                )
                
                idx_min = np.argmin(dist_matrix, axis=1)
                df_work['assigned_rep'] = df_active_v['sales rep'].values[idx_min]
                df_work['dist_km_oneway'] = np.min(dist_matrix, axis=1)
                
                # Distanza round-trip corretta
                df_work['dist_km_round'] = df_work['dist_km_oneway'] * df_work['tortuosity'] * 2
                
                # Calcolo frequenze e ore
                freq_map = {'A': freq_a, 'B': freq_b, 'C': freq_c}
                df_work['freq_visite'] = df_work['classe'].map(freq_map)
                
                # Ore per cliente (CORRETTO: non moltiplicare per numero clienti!)
                df_work['ore_visita_annue'] = (df_work['freq_visite'] * durata_visita_min) / 60
                df_work['ore_viaggio_annue'] = df_work['dist_km_round'] / vel_media
                df_work['ore_totali_cliente'] = df_work['ore_visita_annue'] + df_work['ore_viaggio_annue']
                
                # Aggregazione per venditore
                agg = df_work.groupby('assigned_rep').agg(
                    n_clienti=('assigned_rep', 'count'),
                    n_classe_a=('classe', lambda x: (x=='A').sum()),
                    n_classe_b=('classe', lambda x: (x=='B').sum()),
                    n_classe_c=('classe', lambda x: (x=='C').sum()),
                    volume_totale=(col_volume, 'sum'),
                    ore_visite_annue=('ore_visita_annue', 'sum'),
                    ore_viaggio_annue=('ore_viaggio_annue', 'sum'),
                    ore_totali_annue=('ore_totali_cliente', 'sum'),
                    dist_media_km=('dist_km_oneway', 'mean')
                ).reset_index().rename(columns={'assigned_rep': 'sales_rep'})
                
                # KPI derivati
                agg['ore_disponibili'] = max_ore_annue
                agg['saturazione_pct'] = (agg['ore_totali_annue'] / max_ore_annue) * 100
                agg['driving_min_giorno'] = (agg['ore_viaggio_annue'] * 60) / giorni_lavoro_anno
                agg['visite_giorno'] = (agg['ore_visite_annue'] * 60 / durata_visita_min) / giorni_lavoro_anno
                
                # Stato
                def get_alert(sat):
                    if sat > 110: return "🔴 CRITICO"
                    elif sat > 100: return "🟠 OVERLOAD"
                    elif sat > 85: return "🟡 ATTENZIONE"
                    else: return "🟢 OK"
                agg['stato'] = agg['saturazione_pct'].apply(get_alert)
                
                # Merge con tutti i venditori attivi
                all_reps_df = pd.DataFrame({'sales_rep': active_list})
                result = all_reps_df.merge(agg, on='sales_rep', how='left').fillna(0)
                result = result.sort_values('saturazione_pct', ascending=False)
                
                # Salva in sessione
                st.session_state.result_df = result
                st.session_state.df_work = df_work
                st.session_state.df_active_v = df_active_v
                st.session_state.active_list = active_list
                st.session_state.last_calc = pd.Timestamp.now()
                
            except Exception as e:
                st.error(f"❌ Errore calcolo: {str(e)}")
                st.exception(e)
                return
    
    # Visualizzazione risultati
    if st.session_state.result_df is not None:
        result = st.session_state.result_df
        df_work = st.session_state.df_work
        active_list = st.session_state.active_list
        
        # KPI riepilogo
        st.subheader("📊 KPI Scenario")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Venditori Attivi", len(active_list))
        with col2:
            total_customers = int(result['n_clienti'].sum())
            st.metric("Clienti Serviti", f"{total_customers:,}")
        with col3:
            avg_sat = result['saturazione_pct'].mean()
            st.metric("Saturazione Media", f"{avg_sat:.1f}%")
        with col4:
            overload = len(result[result['saturazione_pct'] > 100])
            st.metric("Venditori Overload", overload, delta_color="inverse")
        
        # Tabella dettagliata
        st.subheader("📋 Dettaglio per Venditore")
        
        # Preparazione tabella display
        display_df = result.copy()
        display_df['visite_mese_medie'] = (display_df['ore_visite_annue'] * 60 / durata_visita_min) / 12
        
        cols_display = {
            'sales_rep': 'Venditore',
            'stato': 'Stato',
            'n_clienti': 'Clienti Totali',
            'n_classe_a': '• A',
            'n_classe_b': '• B',
            'n_classe_c': '• C',
            'volume_totale': f'Volume',
            'ore_visite_annue': 'Ore Visite/anno',
            'ore_viaggio_annue': 'Ore Viaggio/anno',
            'ore_totali_annue': 'Ore Totali',
            'saturazione_pct': 'Saturazione %',
            'driving_min_giorno': 'Driving Min/Giorno',
            'visite_giorno': 'Visite/Giorno'
        }
        
        df_show = display_df[list(cols_display.keys())].copy()
        df_show.columns = [cols_display[c] for c in df_show.columns]
        
        # Conditional formatting
        def color_sat(val):
            if pd.isna(val): return ''
            try:
                num = float(str(val).replace('%','').strip())
                if num > 110: return 'background-color: #ffcdd2; color: #b71c1c'
                elif num > 100: return 'background-color: #ffe0b2; color: #e65100'
                elif num > 85: return 'background-color: #fff9c4; color: #f57f17'
                else: return 'background-color: #c8e6c9; color: #1b5e20'
            except: return ''
        
        styled = df_show.style.map(color_sat, subset=['Saturazione %']).format({
            'Volume': '{:,.0f}',
            'Ore Visite/anno': '{:.1f}',
            'Ore Viaggio/anno': '{:.1f}',
            'Ore Totali': '{:.1f}',
            'Saturazione %': '{:.1f}%',
            'Driving Min/Giorno': '{:.1f}',
            'Visite/Giorno': '{:.2f}'
        })
        
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
        # Mappa
        st.subheader("🗺️ Mappa Territori")
        
        show_hull = st.checkbox("🔷 Mostra confini territori", value=True)
        
        fig = px.scatter_mapbox(
            df_work,
            lat="latitudine", lon="longitudine",
            color="assigned_rep",
            size="freq_visite",
            size_max=8,
            hover_data={
                "classe": True,
                "freq_visite": True,
                "assigned_rep": True,
                "latitudine": False,
                "longitudine": False
            },
            zoom=5, height=600, render_mode="webgl"
        )
        
        # Aggiungi poligoni
        if show_hull:
            colors = px.colors.qualitative.Set3
            for i, rep in enumerate(active_list):
                subset = df_work[df_work['assigned_rep'] == rep]
                lon_hull, lat_hull = compute_convex_hull(subset, 'latitudine', 'longitudine')
                if lon_hull is not None:
                    fig.add_trace(go.Scattermapbox(
                        mode="lines",
                        lon=lon_hull, lat=lat_hull,
                        line=dict(width=2, color=colors[i % len(colors)]),
                        name=f"{rep}",
                        showlegend=True
                    ))
        
        # Marker venditori
        fig.add_trace(go.Scattermapbox(
            lat=st.session_state.df_active_v['latitudine'],
            lon=st.session_state.df_active_v['longitudine'],
            mode='markers+text',
            marker=dict(size=14, symbol='star', color='black', line=dict(width=2, color='white')),
            text=st.session_state.df_active_v['sales rep'],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            name='🏠 Home Base',
            hoverinfo='text'
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":30,"l":0,"b":0},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Dettaglio venditore
        st.subheader("🔍 Dettaglio Operativo")
        selected_rep = st.selectbox("Seleziona venditore", active_list)
        
        if selected_rep:
            rep_data = result[result['sales_rep'] == selected_rep].iloc[0]
            rep_cust = df_work[df_work['assigned_rep'] == selected_rep]
            
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                st.markdown(f"**{selected_rep}**")
                st.markdown(f"- **Stato**: {rep_data['stato']}")
                st.markdown(f"- **Clienti**: {int(rep_data['n_clienti'])} (A:{int(rep_data['n_classe_a'])} B:{int(rep_data['n_classe_b'])} C:{int(rep_data['n_classe_c'])})")
                st.markdown(f"- **Volume**: {rep_data['volume_totale']:,.0f}")
                st.markdown(f"- **Ore Visite/anno**: {rep_data['ore_visite_annue']:.1f}")
                st.markdown(f"- **Ore Viaggio/anno**: {rep_data['ore_viaggio_annue']:.1f}")
                st.markdown(f"- **Saturazione**: {rep_data['saturazione_pct']:.1f}%")
                st.markdown(f"- **Driving/giorno**: {rep_data['driving_min_giorno']:.1f} min")
                st.markdown(f"- **Visite/giorno**: {rep_data['visite_giorno']:.2f}")
                
                if rep_data['driving_min_giorno'] > 180:
                    st.warning(f"⚠️ Driving elevato: {rep_data['driving_min_giorno']:.0f} min/giorno")
            
            with col_d2:
                if len(rep_cust) > 0:
                    fig_zoom = px.scatter_mapbox(
                        rep_cust,
                        lat="latitudine", lon="longitudine",
                        color="classe",
                        color_discrete_map={'A':'red', 'B':'orange', 'C':'green'},
                        size="freq_visite",
                        zoom=7, height=300, render_mode="webgl"
                    )
                    rep_home = st.session_state.df_active_v[st.session_state.df_active_v['sales rep'] == selected_rep]
                    if len(rep_home) > 0:
                        fig_zoom.add_trace(go.Scattermapbox(
                            lat=rep_home['latitudine'], lon=rep_home['longitudine'],
                            mode='markers', marker=dict(size=15, symbol='star', color='black'),
                            name='🏠', showlegend=False
                        ))
                    fig_zoom.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                    st.plotly_chart(fig_zoom, use_container_width=True)
        
        # Export
        st.divider()
        st.subheader("💾 Export Dati")
        
        export_df = result.copy()
        export_df['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        csv = export_df.to_csv(index=False, decimal=';', sep=';')
        
        st.download_button(
            label="📥 Scarica Report CSV",
            data=csv,
            file_name=f"scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.info("👈 Configura parametri e clicca 'CALCOLA SCENARIO'")

if __name__ == "__main__":
    main()
