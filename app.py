import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Ottimizzazione Giri Visite", layout="wide")
st.title("🚚 Sistema di Ottimizzazione Giri Visite")

# 🔍 SEGNALE DI CONTROLLO: Se vedi questa scritta, l'app si è aggiornata!
st.sidebar.markdown("### 🟢 STATO APP: VERSIONE NUOVA BLINDATA")

uploaded_file = st.sidebar.file_uploader("Carica il file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Caricamento fogli
        df_clienti = pd.read_excel(uploaded_file, sheet_name="clienti_geocodificati")
        df_venditori = pd.read_excel(uploaded_file, sheet_name="venditori")
        
        # Pulizia totale dei nomi delle colonne (tutto minuscolo e senza spazi vuoti)
        df_clienti.columns = [str(c).strip().lower() for c in df_clienti.columns]
        df_venditori.columns = [str(c).strip().lower() for c in df_venditori.columns]
        
        # Identificazione colonne chiave con tolleranza totale sui nomi
        col_c_rep = next((c for c in df_clienti.columns if "rep" in c or "venditore" in c or "agente" in c), None)
        col_v_rep = next((c for c in df_venditori.columns if "rep" in c or "venditore" in c or "agente" in c or "etichette" in c), None)
        
        if not col_c_rep or not col_v_rep:
            st.error("❌ Impossibile trovare la colonna del Venditore/Sales Rep nei fogli Excel. Verifica i nomi.")
            st.stop()
            
        # Allineamento nomi interni
        df_clienti = df_clienti.rename(columns={col_c_rep: "sales_rep_key"})
        df_venditori = df_venditori.rename(columns={col_v_rep: "sales_rep_key"})
        
        col_c_nome = next((c for c in df_clienti.columns if "name" in c or "ragione" in c or "cliente" in c), df_clienti.columns[0])
        col_c_lat = next((c for c in df_clienti.columns if "lat" in c), None)
        col_c_lon = next((c for c in df_clienti.columns if "lon" in c), None)
        col_v_lat = next((c for c in df_venditori.columns if "lat" in c), None)
        col_v_lon = next((c for c in df_venditori.columns if "lon" in c), None)
        
        # Funzioni di calcolo e pulizia coordinate
        def haversine(lat1, lon1, lat2, lon2):
            try:
                r = 6371.0
                p1, p2 = np.radians(float(lat1)), np.radians(float(lat2))
                dp = np.radians(float(lat2) - float(lat1))
                dl = np.radians(float(lon2) - float(lon1))
                a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
                return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) * r
            except: return np.nan

        def pulisci_coord(v):
            try:
                val = float(v)
                if val > 90 or val < -90:
                    s = str(int(val))
                    if len(s) >= 5: return float(f"{s[:2]}.{s[2:]}")
                return val
            except: return np.nan

        if col_v_lat and col_v_lon:
            df_venditori[col_v_lat] = df_venditori[col_v_lat].apply(pulisci_coord)
            df_venditori[col_v_lon] = df_venditori[col_v_lon].apply(pulisci_coord)
        if col_c_lat and col_c_lon:
            df_clienti[col_c_lat] = df_clienti[col_c_lat].apply(pulisci_coord)
            df_clienti[col_c_lon] = df_clienti[col_c_lon].apply(pulisci_coord)

        df_v_pulito = df_venditori.dropna(subset=["sales_rep_key"]).drop_duplicates(subset=["sales_rep_key"])
        mappa_v = df_v_pulito.set_index("sales_rep_key")[[col_v_lat, col_v_lon]].to_dict(orient="index")

        distanze = []
        for _, riga in df_clienti.iterrows():
            rep = riga["sales_rep_key"]
            lat_c, lon_c = riga[col_c_lat], riga[col_c_lon]
            if rep in mappa_v and pd.notna(lat_c) and pd.notna(lon_c):
                v_lat, v_lon = mappa_v[rep][col_v_lat], mappa_v[rep][col_v_lon]
                if pd.notna(v_lat) and pd.notna(v_lon):
                    distanze.append(haversine(v_lat, v_lon, lat_c, lon_c))
                    continue
            distanze.append(np.nan)
        df_clienti["distanza_km"] = distanze

        # Analisi ABC dinamica
        colonne_volumi = [c for c in df_clienti.columns if any(x in c for x in ["gy", "du", "co", "tot", "pezzi"])]
        volume_scelto = st.sidebar.selectbox("Colonna volumi per ABC:", options=colonne_volumi if colonne_volumi else [df_clienti.columns[0]])
        
        df_clienti[volume_scelto] = pd.to_numeric(df_clienti[volume_scelto], errors='coerce').fillna(0)
        df_ordinato = df_clienti.sort_values(by=volume_scelto, ascending=False).copy()
        tot = df_ordinato[volume_scelto].sum()
        df_ordinato["cum"] = df_ordinato[volume_scelto].cumsum() / tot * 100 if tot > 0 else 0
        df_ordinato["classe_abc"] = df_ordinato["cum"].apply(lambda x: "A" if x <= 70 else ("B" if x <= 90 else "C"))

        # Visualizzazione metriche e tabelle
        st.columns(2)[0].metric("📊 Clienti Totali", f"{len(df_clienti)} anagrafiche")
        st.columns(2)[1].metric("👤 Venditori Totali", f"{len(df_v_pulito)} sales rep")

        st.subheader("📍 Riepilogo Dati Calcolati")
        st.dataframe(df_ordinato)

        st.subheader("🗺️ Mappa Distribuzione")
        fig = px.scatter_mapbox(df_ordinato, lat=col_c_lat, lon=col_c_lon, color="classe_abc", hover_name=col_c_nome, zoom=5, height=500)
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Qualcosa è andato storto nell'elaborazione: {e}")
else:
    st.info("👋 In attesa del caricamento del file Excel definitivo.")
