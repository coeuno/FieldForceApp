import streamlit as st

st.title("Test Base")
st.write("Se vedi questo, l'app funziona")

# Test import sklearn
try:
    from sklearn.cluster import KMeans
    st.success("✅ sklearn importato")
except Exception as e:
    st.error(f"❌ sklearn non disponibile: {e}")

# Test import scipy
try:
    from scipy.spatial import ConvexHull
    st.success("✅ scipy importato")
except Exception as e:
    st.error(f"❌ scipy non disponibile: {e}")
