import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import yaml

# --- CONFIGURAZIONE INTERFACCIA ---
st.set_page_config(page_title="Strategic Downsizing Simulator", layout="wide")
st.title("📊 Strategic Downsizing Simulator & Force Redistribution")
st.markdown("---")

# --- FUNZIONI DI CACHING E CALCOLO VETTORIALE ---
@st.cache_data
def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Calcolo ultrarapido della distanza in KM tra array di coordinate"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

def calcola_scenari(df_c, df_v, config, active_reps):
    """Motore matematico vettorializzato per la riallocazione e calcolo carichi"""
    # 1. Selezione Metrica e Classificazione ABC
    metric = config['metric']
    df_c['vol_target'] = pd.to_numeric(df_c[metric], errors='coerce').fillna(0)
    
    # Filtro eleggibilità (Taglia Minima)
    df_eligible = df_c[df_c['vol_target'] >= config['min_size']].copy()
    
    # Soglie ABC Dinamiche (basate su valori assoluti impostati nei cursori)
    def assegna_abc(v):
        if v >= config['threshold_a']: return 'A'
        elif v >= config['threshold_b']: return 'B'
        return 'C'
    df_eligible['classe_abc'] = df_eligible['vol_target'].apply(assegna_abc)
    
    # Assegnazione frequenze e ore visita
    freq_map = {'A': config['freq_a'], 'B': config['freq_b'], 'C': config['freq_c']}
    df_eligible['visite_annue'] = df_eligible['classe_abc'].map(freq_map)
    df_eligible['ore_visita_annue'] = df_eligible['visite_annue'] * config['h_visit']
    
    # 2. Algoritmo di Riassegnazione Geografica Auto-Adattiva
    # Identifica i venditori rimasti attivi
    df_v_active = df_v[df_v['sales_rep_key'].isin(active_reps)].copy()
    
    if df_v_active.empty:
        st.error("❌ Errore critico: Selezionare almeno un venditore attivo.")
        st.stop()
        
    # Coordinate dei venditori attivi
    v_coords = df_v_active[['longitudine', 'latitudine']].values # Shape (N_v, 2)
    v_keys = df_v_active['sales_rep_key'].values
    
    # Coordinate dei clienti
    c_coords = df_eligible[['longitudine', 'latitudine']].values # Shape (N_c, 2)
    
    # Matrice delle distanze (Clienti x Venditori) usando Haversine vettoriale broadcastato
    # Estendiamo le dimensioni per sfruttare il broadcasting di NumPy
    c_lon = c_coords[:, 0][:, np.newaxis]
    c_lat = c_coords[:, 1][:, np.newaxis]
    v_lon = v_coords[:, 0][np.newaxis, :]
    v_lat = v_coords[:, 1][np.newaxis, :]
    
    dist_matrix = haversine_vectorized(c_lon, c_lat, v_lon, v_lat)
    
    # Trova l'indice del venditore più vicino per ogni cliente
    min_dist_idx = np.argmin(dist_matrix, axis=1)
    
    # Assegnazione nuovo venditore e distanza relativa
    df_eligible['assigned_rep'] = v_keys[min_dist_idx]
    df_eligible['dist_da_casa_km'] = np.amin(dist_matrix, axis=1)
    
    # Tracciamento riassegnazione vs Baseline originaria
    df_eligible['is_reassigned'] = df_eligible['assigned_rep'] != df_eligible['sales_rep_key']
    
    # 3. Calcolo dei tempi di viaggio tramite Proxy Geometrico
    # Ore viaggio = (Distanza Casa-Base * Coeff Tortuosità * 2 per Andata/Ritorno * Frequenza) / Velocità Media
    df_eligible['ore_viaggio_annue'] = (
        (df_eligible['dist_da_casa_km'] * config['tortuosity'] * 2 * df_eligible['visite_annue']) 
        / config['avg_speed']
    )
    
    return df_eligible, df_v_active

# --- CARICAMENTO FILE DA SIDEBAR ---
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Carica File Excel Unificato", type=["xlsx"])

if uploaded_file is not None:
    # Lettura dei fogli standardizzati
    try:
        df_clienti_grezzo = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
        df_venditori_grezzo = pd.read_excel(uploaded_file, sheet_name="venditori")
        
        df_clienti_grezzo.columns = [str(c).strip().lower() for c in df_clienti_grezzo.columns]
        df_venditori_grezzo.columns = [str(c).strip().lower() for c in df_venditori_grezzo.columns]
        
        col_c_rep = next((c for c in df_clienti_grezzo.columns if "rep" in c or "venditore" in c or "agente" in c), "sales rep")
        col_v_rep = next((c for c in df_venditori_grezzo.columns if "rep" in c or "venditore" in c or "agente" in c or "etichette" in c), "sales rep")
        
        df_c = df_clienti_grezzo.rename(columns={col_c_rep: "sales_rep_key"}).copy()
        df_v = df_venditori_grezzo.rename(columns={col_v_rep: "sales_rep_key"}).copy()
        
        # Pulizia rigorosa coordinate
        df_c = df_c.dropna(subset=['latitudine', 'longitudine'])
        df_v = df_v.dropna(subset=['latitudine', 'longitudine']).drop_duplicates(subset=['sales_rep_key'])
        
        # Lista di tutte le risorse disponibili (Baseline fissa a 23 risorse)
        tutti_venditori = sorted(df_v['sales_rep_key'].unique().tolist())
        
    except Exception as e:
        st.error(f"Errore nella struttura del file Excel: {e}")
        st.stop()

    # --- PANNELLO CONFIGURAZIONE E PARAMETRI (SIDEBAR) ---
    st.sidebar.header("🎛️ Parametri di Scenario")
    
    # 1. Configurazione clusterizzazione vendite
    colonne_volumi = [c for c in df_c.columns if any(x in c for x in ["gy", "du", "co", "tot", "25", "26"])]
    metric_sel = st.sidebar.selectbox("Metrica Volume di Riferimento", options=colonne_volumi, index=colonne_volumi.index('tot 26') if 'tot 26' in colonne_volumi else 0)
    
    max_val_metric = int(df_c[metric_sel].max())
    min_size = st.sidebar.slider("Taglia Minima Cliente (Esclusione Low-Value)", 0, 100, 0)
    
    t_a = st.sidebar.slider("Soglia Minima Classe A (Pezzi/Anno)", 0, max_val_metric, int(max_val_metric*0.4))
    t_b = st.sidebar.slider("Soglia Minima Classe B (Pezzi/Anno)", 0, t_a, int(max_val_metric*0.1))
    
    # 2. Configurazione Frequenze e Ore Lavoro
    st.sidebar.subheader("⏱️ Carichi di Lavoro & Logistica")
    f_a = st.sidebar.number_input("Visite/Anno Classe A (2 al mese = 24)", value=24)
    f_b = st.sidebar.number_input("Visite/Anno Classe B (1 al mese = 12)", value=12)
    f_c = st.sidebar.number_input("Visite/Anno Classe C (1 ogni 2 mesi = 6)", value=6)
    
    h_visit = st.sidebar.number_input("Durata Singola Visita (Ore)", value=1.5, step=0.1)
    max_hours = st.sidebar.number_input("Tetto Ore Lavorative Annue/Capacità", value=1760)
    avg_speed = st.sidebar.slider("Velocità Media di Trasferimento (KM/H)", 40, 110, 70)
    tortuosity = st.sidebar.slider("Coefficiente di Tortuosità Stradale", 1.0, 1.5, 1.2, step=0.05)

    # Raccolta configurazione corrente in dizionario
    current_config = {
        'metric': metric_sel, 'min_size': min_size, 'threshold_a': t_a, 'threshold_b': t_b,
        'freq_a': f_a, 'freq_b': f_b, 'freq_c': f_c, 'h_visit': h_visit,
        'max_hours': max_hours, 'avg_speed': avg_speed, 'tortuosity': tortuosity
    }

    # --- 2. SISTEMA SALVA/CARICA SCENARIO IN YAML ---
    st.sidebar.subheader("💾 Gestione Scenari YAML")
    
    # Pannello Selezione Organico ( Checklist ON/OFF per i 23 venditori)
    st.sidebar.subheader("👤 Stato Forza Vendita (ON / OFF)")
    active_reps = []
    
    # Caricamento file YAML esterno opzionale
    uploaded_yaml = st.sidebar.file_uploader("Carica configurazione (.yaml)", type=["yaml"])
    if uploaded_yaml is not None:
        try:
            loaded_scen = yaml.safe_load(uploaded_yaml)
            # Sovrascrittura dinamica delle spunte se presente nel file caricato
            saved_reps = loaded_scen.get('active_vendors', tutti_venditori)
            for v in tutti_venditori:
                is_on = st.sidebar.checkbox(f"🟢 {v}", value=(v in saved_reps), key=f"v_{v}")
                if is_on: active_reps.append(v)
            # Aggiornamento parametri da file yaml se necessario
            current_config.update({k: v for k, v in loaded_scen.items() if k != 'active_vendors'})
        except Exception as e:
            st.error(f"Errore nel parsing del file YAML: {e}")
    else:
        # Default: Gestione spunte manuale
        for v in tutti_venditori:
            is_on = st.sidebar.checkbox(f"🟢 {v}", value=True, key=f"v_{v}")
            if is_on: active_reps.append(v)

    # Esportazione scenario corrente
    current_config['active_vendors'] = active_reps
    yaml_string = yaml.dump(current_config, default_flow_style=False)
    st.sidebar.download_button(
        label="📥 Esporta Scenario Corrente (.yaml)",
        data=yaml_string,
        file_name="scenario_ristrutturazione.yaml",
        mime="application/x-yaml"
    )

    # --- ESECUZIONE MOTORE DI CALCOLO ---
    df_output_clienti, df_v_attivi = calcola_scenari(df_c, df_v, current_config, active_reps)

    # Aggregazione metriche per Venditore Superstite
    df_rep_metrics = df_output_clienti.groupby('assigned_rep').agg(
        clienti_tot=('payer id', 'count'),
        clienti_assorbiti=('is_reassigned', 'sum'),
        volume_gestito=('vol_target', 'sum'),
        ore_visite=('ore_visita_annue', 'sum'),
        ore_viaggio=('ore_viaggio_annue', 'sum')
    ).reset_index()
    
    df_rep_metrics['ore_totali_richieste'] = df_rep_metrics['ore_visite'] + df_rep_metrics['ore_viaggio']
    df_rep_metrics['saturazione_percentuale'] = (df_rep_metrics['ore_totali_richieste'] / current_config['max_hours']) * 100

    # --- 3. MODULO VALIDAZIONE POST-CALCOLO (KPI DIREZIONALI) ---
    st.subheader("📊 Modulo di Validazione Post-Calcolo (Scenario KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    headcount_attivo = len(active_reps)
    pct_downsized = ((23 - headcount_attivo) / 23) * 100
    kpi1.metric("📉 Riduzione Organico", f"{headcount_attivo} / 23 risorse", f"-{pct_downsized:.1f}%")
    
    # Bilanciamento del carico di lavoro (Deviazione Standard del Workload)
    workload_std = df_rep_metrics['ore_totali_richieste'].std()
    workload_mean = df_rep_metrics['ore_totali_richieste'].mean()
    std_dev_pct = (workload_std / workload_mean * 100) if workload_mean > 0 else 0
    kpi2.metric("⚖️ Sbilanciamento Carico (StdDev %)", f"{std_dev_pct:.1f}%", help="Valori alti indicano territori mal distribuiti tra i superstiti.")
    
    # Percentuale Clienti Riassegnati vs Portafoglio Iniziale
    clienti_totali_eleggibili = len(df_output_clienti)
    clienti_riassegnati = df_output_clienti['is_reassigned'].sum()
    pct_riassegnati = (clienti_riassegnati / clienti_totali_eleggibili * 100) if clienti_totali_eleggibili > 0 else 0
    kpi3.metric("🔄 Portafoglio Clienti Riassegnato", f"{pct_riassegnati:.1f}%", f"{clienti_riassegnati} p.v. spostati")
    
    # Stima massima ore guida giornaliere simulate sul venditore più sotto pressione
    # Assumendo 220 giorni lavorativi annui
    max_yearly_travel_hours = df_rep_metrics['ore_viaggio'].max()
    max_daily_driving = (max_yearly_travel_hours / 220) if pd.notna(max_yearly_travel_hours) else 0
    kpi4.metric("🚗 Stima Max Guida Giornaliera", f"{max_daily_driving:.1f} ore/giorno", help="Basato sul commerciale con l'area geografica più estesa.")

    # --- SEZIONE WARNINGS / ALERTER OPERATIVI ---
    overloaded_reps = df_rep_metrics[df_rep_metrics['ore_totali_richieste'] > current_config['max_hours']]
    if not overloaded_reps.empty:
        st.error(f"⚠️ **ALLERTA DIREZIONE:** Ci sono {len(overloaded_reps)} risorse in stato di **Overload Critico** (> {current_config['max_hours']} ore/anno). Lo scenario calcolato è insostenibile.")
    else:
        st.success("✅ **VALIDAZIONE SCENARIO:** Tutte le risorse superstiti rientrano nei tetti di capacità lavorativa impostati.")

    # --- 1. FUNZIONE PLOT_MAP() CON LAYERS E CONVEX HULLS ---
    def plot_map(df_c, df_v_act):
        st.subheader("🗺️ Mappatura Spaziale e Poligoni di Competenza")
        
        # Generazione Mappa Scatter WebGL dei clienti eleggibili
        fig = px.scatter_mapbox(
            df_c, lat="latitudine", lon="longitudine",
            color="assigned_rep", category_orders={"assigned_rep": active_reps},
            hover_name="payer name", 
            hover_data={"vol_target": True, "classe_abc": True, "sales_rep_key": True, "assigned_rep": True},
            zoom=5, height=650, render_mode="webgl"
        )
        
        # Aggiunta Layer Poligoni Convex Hull per ogni Venditore Superstite
        for rep in df_v_act['sales_rep_key'].unique():
            rep_data = df_c[df_c['assigned_rep'] == rep]
            
            # Un poligono richiede almeno 3 punti non allineati per essere calcolato
            if len(rep_data) >= 3:
                points = rep_data[['longitudine', 'latitudine']].values
                try:
                    hull = ConvexHull(points)
                    # Chiusura del poligono ripetendo il primo punto alla fine
                    hull_points = points[hull.vertices]
                    hull_lon = hull_points[:, 0].tolist() + [hull_points[0, 0]]
                    hull_lat = hull_points[:, 1].tolist() + [hull_points[0, 1]]
                    
                    # Tracciamento area ombreggiata trasparente su Plotly
                    fig.add_trace(go.Scattermapbox(
                        lon=hull_lon, lat=hull_lat,
                        mode='lines', fill='toself',
                        fillcolor='rgba(100,100,100,0.15)', # Trasparenza per sovrapposizioni
                        line=dict(width=1.5),
                        name=f"Confini Area {rep}",
                        legendgroup=f"group_{rep}",
                        showlegend=False
                    ))
                except:
                    # Gestione eccezione se i punti geometrici sono collineari
                    pass
                    
        # Layer Case Venditori (Punti di Origine Geografica)
        fig.add_trace(go.Scattermapbox(
            lon=df_v_act['longitudine'], lat=df_v_act['latitudine'],
            mode='markers+text',
            marker=dict(size=12, color='black', symbol='circle'),
            text=df_v_act['sales_rep_key'], textposition="top center",
            name="📍 Residenze Venditori ON"
        ))
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(title="Forza Vendita Attiva")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Chiamata alla funzione Mappa
    plot_map(df_output_clienti, df_v_attivi)

    # --- TABELLA DI VALIDAZIONE SCENARIO PER IL BOARD ---
    st.subheader("📋 Tabella Comparativa di Saturazione e Impatto Volumi")
    
    # Formattazione per visualizzazione pulita delle metriche
    df_view = df_rep_metrics.rename(columns={
        'assigned_rep': 'SALES REP SUPERSTITE',
        'clienti_tot': 'N° CLIENTI PV',
        'clienti_assorbiti': 'CLIENTI ASSORBITI',
        'volume_gestito': 'VOLUME GESTITO (PZ)',
        'ore_totali_richieste': 'ORE TOTALI RICHIESTE',
        'saturazione_percentuale': 'SATURAZIONE (%)'
    })
    
    st.dataframe(
        df_view[['SALES REP SUPERSTITE', 'N° CLIENTI PV', 'CLIENTI ASSORBITI', 'VOLUME GESTITO (PZ)', 'ORE TOTALI RICHIESTE', 'SATURAZIONE (%)']]
        .style.format({
            'VOLUME GESTITO (PZ)': '{:,.0f}',
            'ORE TOTALI RICHIESTE': '{:,.1f} h',
            'SATURAZIONE (%)': '{:.1f} %'
        }).background_gradient(subset=['SATURAZIONE (%)'], cmap="YlOrRd", vmin=50, vmax=100)
    )

else:
    st.info("👋 In attesa del caricamento del file Excel aziendale unificato per avviare la simulazione di downsizing.")
