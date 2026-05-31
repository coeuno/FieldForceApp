import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from geopy.distance import geodesic

# Configurazione della pagina
st.set_page_config(page_title="Ottimizzazione Giri Visite", layout="wide")

st.title("🚚 Sistema di Ottimizzazione Giri Visite Sales Rep")
st.write("Carica il tuo file Excel definitivo con i fogli 'clienti_geocodificati' e 'venditori' per analizzare le rotte.")

# --- BARRA LATERALE PER IL CARICAMENTO DATI ---
st.sidebar.header("📂 Caricamento File Definitivo")
uploaded_file = st.sidebar.file_uploader(
    "Carica il file Excel (.xlsx)", 
    type=["xlsx"],
    help="Seleziona il file che contiene i fogli 'clienti_geocodificati' e 'venditori'"
)

# Funzione di correzione coordinate per Excel Italiano (es. 446471 -> 44.6471)
def aggiusta_coordinate(valore):
    try:
        val = float(valore)
        if val > 90 or val < -90: # Se non ha i decimali corretti ed è enorme
            # Trova quante cifre ha e sposta il decimale dopo le prime due cifre
            s_val = str(int(val))
            if len(s_val) >= 5:
                return float(f"{s_val[:2]}.{s_val[2:]}")
        return val
    except:
        return np.nan

if uploaded_file is not None:
    try:
        # Lettura automatica dei due fogli dal file unico Excel
        df_clienti = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
        df_venditori_grezzo = pd.read_excel(uploaded_file, sheet_name="venditori")
        
        # Pulizia foglio venditori (prendiamo solo una riga univoca per venditore come hai fatto nella Pivot)
        df_venditori = df_venditori_grezzo.drop_duplicates(subset=["SALES REP"]).copy()
        
        # Applicazione correzione automatica coordinate venditori
        df_venditori["Latitudine"] = df_venditori["Latitudine"].apply(aggiusta_coordinate)
        df_venditori["Longitudine"] = df_venditori["Longitudine"].apply(aggiusta_coordinate)
        
        # Conteggi reali calcolati analizzando il file
        num_clienti = len(df_clienti)
        num_venditori = len(df_venditori)
        
        # --- KPI LIVE NELL'INTERFACCIA ---
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📊 Clienti Totali Rilevati", value=f"{num_clienti} anagrafiche")
        with col2:
            st.metric(label="👤 Venditori Unici Rilevati", value=f"{num_venditori} sales rep")
            
        # --- SELEZIONE DINAMICA DEL VOLUME ---
        st.sidebar.subheader("📈 Selezione Parametri Volumi")
        colonne_volumi = [c for c in df_clienti.columns if any(x in c for x in ["GY", "DU", "CO", "TOT", "Pezzi"])]
        
        if colonne_volumi:
            volume_scelto = st.sidebar.selectbox(
                "Quale colonna di volumi vuoi usare per il calcolo delle Classi ABC?",
                options=colonne_volumi,
                index=colonne_volumi.index("TOT 25") if "TOT 25" in colonne_volumi else 0
            )
        else:
            # Fallback se non trova le colonne specifiche
            df_clienti["Volume_Default"] = 1
            volume_scelto = "Volume_Default"
            st.sidebar.warning("Nessuna colonna volumi standard identificata. Verrà conteggiato 1 pezzo a cliente.")

        # --- CALCOLO CLASSI ABC DINAMICHE ---
        st.subheader(f"📊 Classificazione ABC Clienti (Basata su colonna: {volume_scelto})")
        
        # Pulizia dati volume
        df_clienti[volume_scelto] = pd.to_numeric(df_clienti[volume_scelto], errors='coerce').fillna(0)
        df_ordinato = df_clienti.sort_values(by=volume_scelto, ascending=False).copy()
        
        totale_pezzi = df_ordinato[volume_scelto].sum()
        df_ordinato["Perc_Cumulata"] = df_ordinato[volume_scelto].cumsum() / totale_pezzi * 100 if totale_pezzi > 0 else 0
        
        def assegna_classe(perc):
            if perc <= 70: return "A"
            elif perc <= 90: return "B"
            else: return "C"
            
        df_ordinato["Classe_ABC"] = df_ordinato["Perc_Cumulata"].apply(assegna_classe)
        
        # Mostra riepilogo classi ABC
        riepilogo_abc = df_ordinato["Classe_ABC"].value_counts().reindex(["A", "B", "C"]).fillna(0)
        c_a, c_b, c_c = st.columns(3)
        c_a.info(f"**Classe A (Top 70% Volumi):** {int(riepilogo_abc['A'])} clienti")
        c_b.warning(f"**Classe B (Centro 20% Volumi):** {int(riepilogo_abc['B'])} clienti")
        c_c.error(f"**Classe C (Coda 10% Volumi):** {int(riepilogo_abc['C'])} clienti")

        # --- CALCOLO DISTANZE REALI CASA-CLIENTE ---
        st.subheader("📍 Verifica Distanze e Assegnazioni")
        
        # Creiamo un dizionario con le coordinate dei venditori per cercarle velocemente
        mappa_venditori = df_venditori.set_index("SALES REP")[["Latitudine", "Longitudine"]].to_dict(orient="index")
        
        distanze_km = []
        for idx, riga in df_ordinato.iterrows():
            rep = riga["SALES REP"]
            lat_c, lon_c = riga["Latitudine"], riga["Longitudine"]
            
            if rep in mappa_venditori and pd.notna(lat_c) and pd.notna(lon_c):
                coord_v = (mappa_venditori[rep]["Latitudine"], mappa_venditori[rep]["Longitudine"])
                coord_c = (lat_c, lon_c)
                try:
                    distanze_km.append(geodesic(coord_v, coord_c).km)
                except:
                    distanze_km.append(np.nan)
            else:
                distanze_km.append(np.nan)
                
        df_ordinato["Distanza_da_Casa_KM"] = distanze_km
        
        # Vista tabella pulita dei clienti con distanze reali e classi ABC
        st.dataframe(
            df_ordinato[[
                "SALES REP", "SOLD TO NAME", "COMUNE", "PROVINCIA", 
                volume_scelto, "Classe_ABC", "Distanza_da_Casa_KM"
            ]].style.format({"Distanza_da_Casa_KM": "{:.1f} km", volume_scelto: "{:,.0f}"})
        )
        
        # --- MAPPA INTERATTIVA ---
        st.subheader("🗺️ Mappa Distribuzione Clienti e Venditori")
        # Prepariamo i dati per la mappa di Plotly
        df_ordinato["Tipo"] = "Cliente " + df_ordinato["Classe_ABC"]
        df_venditori["Tipo"] = "Venditore (Casa)"
        df_venditori["SOLD TO NAME"] = df_venditori["SALES REP"]
        df_venditori["COMUNE"] = df_venditori["Abitazione"]
        df_venditori[volume_scelto] = 0
        
        mappa_df = pd.concat([
            df_ordinato[["Latitudine", "Longitudine", "SOLD TO NAME", "COMUNE", "Tipo", volume_scelto]],
            df_venditori[["Latitudine", "Longitudine", "SOLD TO NAME", "COMUNE", "Tipo", volume_scelto]]
        ]).dropna(subset=["Latitudine", "Longitudine"])
        
        fig = px.scatter_mapbox(
            mappa_df,
            lat="Latitudine",
            lon="Longitudine",
            color="Tipo",
            hover_name="SOLD TO NAME",
            hover_data=["COMUNE", volume_scelto],
            zoom=5,
            height=600,
            color_discrete_map={
                "Cliente A": "#1f77b4",
                "Cliente B": "#ff7f0e",
                "Cliente C": "#d62728",
                "Venditore (Casa)": "#2ca02c"
            }
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Errore nel caricamento del file. Verifica che i nomi dei fogli siano corretti. Dettaglio: {e}")
else:
    st.info("👋 In attesa del caricamento del file Excel definitivo nella barra laterale.")
