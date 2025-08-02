import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="🛍️ E-Commerce Recommendation App", layout="centered")

# --- DOWNLOAD LARGE FILE IF NEEDED ---
ITEM_SIM_PATH = "item_similarity.pkl"
GDRIVE_FILE_ID = "1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n"
if not os.path.exists(ITEM_SIM_PATH):
    with st.spinner("Downloading large similarity matrix from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", ITEM_SIM_PATH, quiet=False)

# --- LOAD MODELS ---
with open("rfm_kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

item_sim_df = pd.read_pickle(ITEM_SIM_PATH)

# --- APP HEADER ---
st.title("🛍️ E-Commerce Recommendation & Customer Segmentation")
st.markdown("Predict customer segment using **RFM** and recommend products using **item-based collaborative filtering**.")

# --- TABS: Product Recommendation | Customer Segmentation ---
tab1, tab2 = st.tabs(["🎯 Product Recommendation", "👤 Customer Segmentation"])

# ========== TAB 1: PRODUCT RECOMMENDATION ==========
with tab1:
    st.subheader("🔎 Recommend Similar Products")
    product_name = st.selectbox("Enter or select a product you like:", item_sim_df.index.tolist())

    def recommend_items(product_name, top_n=5):
        if product_name not in item_sim_df.columns:
            return []
        scores = item_sim_df[product_name].sort_values(ascending=False)
        return scores.iloc[1:top_n + 1].index.tolist()

    if st.button("🔍 Get Recommendations"):
        recs = recommend_items(product_name)
        if recs:
            st.success(f"Top 5 products similar to **{product_name}**:")
            for i, item in enumerate(recs, 1):
                st.markdown(f"**{i}.** {item}")
        else:
            st.warning("No similar products found.")

# ========== TAB 2: CUSTOMER SEGMENTATION ==========
with tab2:
    st.subheader("📊 Predict Customer Segment")
    with st.form("rfm_form"):
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=60)
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
        monetary = st.number_input("Monetary (total amount spent)", min_value=1, value=500)
        submitted = st.form_submit_button("Predict Segment")

    if submitted:
        rfm_df = pd.DataFrame([[recency, frequency, monetary]], columns=["Recency", "Frequency", "Monetary"])
        rfm_scaled = scaler.transform(rfm_df)
        segment = int(kmeans.predict(rfm_scaled)[0])
        st.success(f"🎯 This customer belongs to **Segment {segment}**.")

# --- FOOTER ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | RFM Segmentation + Item-Based Recommendations")
