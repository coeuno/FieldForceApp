import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configurazione della pagina
st.set_page_config(page_title="Ottimizzazione Giri Visite", layout="wide")

st.title("🚚 Sistema di Ottimizzazione Giri Visite Sales Rep")
st.write("Carica il tuo file Excel definitivo con i fogli 'clienti_geocodificati' e 'venditori' per analizzare le rotte.")

# --- FUNZIONI DI SERVIZIO COSTRUITE IN CASA ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcola la distanza in KM tra due punti geografici (Formula Haversine)"""
    R = 6371.0 # Raggio della Terra in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def aggiusta_coordinate(valore):
    """Corregge il problema dei punti/virgole di Excel Italiano (es. 446471 -> 44.6471)"""
    try:
        val = float(valore)
        if val > 90 or val < -90:
            s_val = str(int(val))
            if len(s_val) >= 5:
                return float(f"{s_val[:2]}.{s_val[2:]}")
        return val
    except:
        return np.nan

# --- BARRA LATERALE PER IL CARICAMENTO DATI ---
st.sidebar.header("📂 Caricamento File Definitivo")
uploaded_file = st.sidebar.file_uploader("Carica il file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # 1. LETTURA DEI FOGLI
        df_clienti_grezzo = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
        df_venditori_grezzo = pd.read_excel(uploaded_file, sheet_name="venditori")
        
        # 2. RICONOSCIMENTO AUTOMATICO COLONNE VENDITORI (Risolve il problema della Pivot)
        col_v_nome = next((c for c in df_venditori_grezzo.columns if any(x in str(c).lower() for x in ["sales", "rep", "agente", "venditore", "etichette", "row"])), None)
        col_v_lat = next((c for c in df_venditori_grezzo.columns if "lat" in str(c).lower()), None)
        col_v_lon = next((c for c in df_venditori_grezzo.columns if "lon" in str(c).lower()), None)
        col_v_comune = next((c for c in df_venditori_grezzo.columns if any(x in str(c).lower() for x in ["abitazione", "città", "citta", "residenza", "dove", "comune"])), None)

        # 3. RICONOSCIMENTO AUTOMATICO COLONNE CLIENTI
        col_c_nome = next((c for c in df_clienti_grezzo.columns if any(x in str(c).lower() for x in ["sold to name", "ragione", "nome", "cliente"])), None)
        col_c_rep = next((c for c in df_clienti_grezzo.columns if any(x in str(c).lower() for x in ["sales", "rep", "agente", "venditore"])), None)
        col_c_lat = next((c for c in df_clienti_grezzo.columns if "lat" in str(c).lower()), None)
        col_c_lon = next((c for c in df_clienti_grezzo.columns if "lon" in str(c).lower()), None)
        col_c_comune = next((c for c in df_clienti_grezzo.columns if any(x in str(c).lower() for x in ["comune", "città", "citta"])), None)
        col_c_prov = next((c for c in df_clienti_grezzo.columns if any(x in str(c).lower() for x in ["provincia", "sigla", "prov"])), None)

        # Uniformiamo i fogli rinominandoli internamente
        df_venditori = df_venditori_grezzo.rename(columns={col_v_nome: "SALES REP", col_v_lat: "Latitudine", col_v_lon: "Longitudine"}).copy()
        df_venditori["Abitazione"] = df_venditori_grezzo[col_v_comune] if col_v_comune else "Non Specificata"
        
        df_clienti = df_clienti_grezzo.rename(columns={
            col_c_nome: "SOLD TO NAME", col_c_rep: "SALES REP", 
            col_c_lat: "Latitudine", col_c_lon: "Longitudine",
            col_c_comune: "COMUNE", col_c_prov: "PROVINCIA"
        }).copy()

        # Pulizia dati venditori (rimozione duplicati e correzione coordinate numeriche enormi)
        df_venditori = df_venditori.dropna(subset=["SALES REP"]).drop_duplicates(subset=["SALES REP"]).copy()
        df_venditori["Latitudine"] = df_venditori["Latitudine"].apply(aggiusta_coordinate)
        df_venditori["Longitudine"] = df_venditori["Longitudine"].apply(aggiusta_coordinate)

        # Mostra i contatori reali
        st.columns(2)[0].metric("📊 Clienti Totali Rilevati", f"{len(df_clienti)} anagrafiche")
        st.columns(2)[1].metric("👤 Venditori Unici Rilevati", f"{len(df_venditori)} sales rep")

        # --- SELEZIONE DINAMICA DEL VOLUME ---
        st.sidebar.subheader("📈 Selezione Parametri Volumi")
        colonne_volumi = [c for c in df_clienti.columns if any(x in str(c).upper() for x in ["GY", "DU", "CO", "TOT", "PEZZI"])]
        
        if colonne_volumi:
            volume_scelto = st.sidebar.selectbox("Quale colonna di volumi usi per le Classi ABC?", options=colonne_volumi, index=colonne_volumi.index("TOT 25") if "TOT 25" in colonne_volumi else 0)
        else:
            df_clienti["Volume_Finto"] = 1
            volume_scelto = "Volume_Finto"

        # --- CALCOLO CLASSI ABC ---
        st.subheader(f"📊 Classificazione ABC Clienti (Su colonna: {volume_scelto})")
        df_clienti[volume_scelto] = pd.to_numeric(df_clienti[volume_scelto], errors='coerce').fillna(0)
        df_ordinato = df_clienti.sort_values(by=volume_scelto, ascending=False).copy()
        
        totale_pezzi = df_ordinato[volume_scelto].sum()
        df_ordinato["Perc_Cumulata"] = df_ordinato[volume_scelto].cumsum() / totale_pezzi * 100 if totale_pezzi > 0 else 0
        df_ordinato["Classe_ABC"] = df_ordinato["Perc_Cumulata"].apply(lambda x: "A" if x <= 70 else ("B" if x <= 90 else "C"))
        
        abc_counts = df_ordinato["Classe_ABC"].value_counts().reindex(["A", "B", "C"]).fillna(0)
        ca, cb, cc = st.columns(3)
        ca.info(f"**Classe A (Top 70% Volumi):** {int(abc_counts['A'])} clienti")
        cb.warning(f"**Classe B (Centro 20% Volumi):** {int(abc_counts['B'])} clienti")
        cc.error(f"**Classe C (Coda 10% Volumi):** {int(abc_counts['C'])} clienti")

        # --- CALCOLO DISTANZE REALI CASA-CLIENTE ---
        st.subheader("📍 Verifica Distanze e Assegnazioni")
        mappa_v = df_venditori.set_index("SALES REP")[["Latitudine", "Longitudine"]].to_dict(orient="index")
        
        distanze = []
        for _, riga in df_ordinato.iterrows():
            rep = riga["SALES REP"]
            lat_c, lon_c = riga["Latitudine"], riga["Longitudine"]
            if rep in mappa_v and pd.notna(lat_c) and pd.notna(lon_c) and pd.notna(mappa_v[rep]["Latitudine"]):
                distanze.append(haversine_distance(mappa_v[rep]["Latitudine"], mappa_v[rep]["Longitudine"], lat_c, lon_c))
            else:
                distanze.append(np.nan)
                
        df_ordinato["Distanza_da_Casa_KM"] = distanze
        
        st.dataframe(df_ordinato[["SALES REP", "SOLD TO NAME", "COMUNE", "PROVINCIA", volume_scelto, "Classe_ABC", "Distanza_da_Casa_KM"]].style.format({"Distanza_da_Casa_KM": "{:.1f} km", volume_scelto: "{:,.0f}"}))
        
        # --- MAPPA INTERATTIVA ---
        st.subheader("🗺️ Mappa Distribuzione Clienti e Venditori")
        df_ordinato["Tipo"] = "Cliente " + df_ordinato["Classe_ABC"]
        df_venditori["Tipo"] = "Venditore (Casa)"
        df_venditori["SOLD TO NAME"] = df_venditori["SALES REP"]
        df_venditori["COMUNE"] = df_venditori["Abitazione"]
        df_venditori[volume_scelto] = 0
        
        mappa_df = pd.concat([df_ordinato[["Latitudine", "Longitudine", "SOLD TO NAME", "COMUNE", "Tipo", volume_scelto]], df_venditori[["Latitudine", "Longitudine", "SOLD TO NAME", "COMUNE", "Tipo", volume_scelto]]]).dropna(subset=["Latitudine", "Longitudine"])
        
        fig = px.scatter_mapbox(mappa_df, lat="Latitudine", lon="Longitudine", color="Tipo", hover_name="SOLD TO NAME", hover_data=["COMUNE", volume_scelto], zoom=5, height=600, color_discrete_map={"Cliente A": "#1f77b4", "Cliente B": "#ff7f0e", "Cliente C": "#d62728", "Venditore (Casa)": "#2ca02c"})
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Si è verificato un problema nella lettura del file Excel: {e}")
        st.info("Verifica che i fogli si chiamino esattamente 'clienti_geocodificati' e 'venditori'.")
else:
    st.info("👋 In attesa del caricamento del file Excel definitivo.")
