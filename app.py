import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
import zipfile
from io import BytesIO

# ----------------- SETUP -----------------
st.set_page_config(page_title="🛍️ Product Recommender & RFM Segmentor", layout="centered")

# ----------------- LOAD FILE FROM GDRIVE -----------------
@st.cache_resource
def download_and_load_similarity():
    file_id = "1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    with open("item_similarity.pkl", "wb") as f:
        f.write(response.content)
    return pd.read_pickle("item_similarity.pkl")

@st.cache_resource
def load_models():
    with open('rfm_kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

@st.cache_data
def load_product_names():
    return pd.read_csv("product_metadata.csv")  # Must have ProductID, ProductName

# ----------------- LOAD ALL DATA -----------------
item_sim_df = download_and_load_similarity()
kmeans, scaler = load_models()
product_df = load_product_names()
product_map = dict(zip(product_df['ProductID'].astype(str), product_df['ProductName']))
name_to_id = {v: k for k, v in product_map.items()}

# ----------------- APP UI -----------------
st.title("🛍️ E-Commerce Product Recommender & Segmentor")

tab1, tab2 = st.tabs(["🎯 Recommend Products", "👤 RFM Segmentation"])

# ----------------- TAB 1: RECOMMENDATION -----------------
with tab1:
    st.subheader("🎯 Recommend Similar Products")
    product_name = st.selectbox("Select a product name:", sorted(name_to_id.keys()))
    
    if st.button("🔍 Show Recommendations"):
        product_id = name_to_id[product_name]
        if str(product_id) not in item_sim_df.columns:
            st.error("No similar products found.")
        else:
            scores = item_sim_df[str(product_id)].sort_values(ascending=False)
            top_ids = scores.index[1:6]
            st.success("Top 5 Recommendations:")
            for pid in top_ids:
                name = product_map.get(str(pid), f"Product {pid}")
                st.markdown(f"- **{name}** (Similarity: {scores[pid]:.2f})")

# ----------------- TAB 2: RFM SEGMENTATION -----------------
with tab2:
    st.subheader("📊 Predict Customer Segment")

    r = st.number_input("Recency (days since last purchase)", min_value=0)
    f = st.number_input("Frequency (number of purchases)", min_value=0)
    m = st.number_input("Monetary (total spent)", min_value=0)

    if st.button("📈 Predict Segment"):
        input_data = np.array([[r, f, m]])
        scaled = scaler.transform(input_data)
        cluster = kmeans.predict(scaled)[0]
        st.success(f"Predicted RFM Cluster: **Segment {cluster}**")

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | RFM Segmentation + Item Similarity Recommendation")
