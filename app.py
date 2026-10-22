import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import *

st.title("Introduction Agglomerative Clustering")

st.subheader("Dataframe")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(file)
st.dataframe(df, use_container_width=True)

st.subheader("Preprocessing")
features = st.multiselect("Choose features set", df.columns.tolist())

if not features:
    st.stop()

X = preprocessing(df, features)
st.dataframe(X, use_container_width=True)

st.subheader("Modeling")
k = st.slider("Value of K (n of clusters)", 2, 10)
linkage = st.selectbox("Choose one linkage", ["ward", "complete", "average", "single"])

labels = clustering(X, k, linkage)
sil, dav = metrics(X, labels)

st.metric("Silhouette Score", f"{sil:.3f}")
st.metric("Davies-Bouldin Index", f"{dav:.3f}")