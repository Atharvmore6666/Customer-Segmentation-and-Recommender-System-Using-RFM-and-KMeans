import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import gdown
import os

st.set_page_config(page_title="🛍️ E-Commerce Recommender & Segmenter", layout="wide")
st.title("🛍️ E-Commerce Recommendation and Segmentation App")

# Load Data and Models
@st.cache_data
def load_metadata():
    return pd.read_csv("product_metadata.csv")

@st.cache_data
def download_and_load_similarity():
    url = "https://drive.google.com/uc?id=1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n"
    output = "item_similarity.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return pd.read_pickle(output)

@st.cache_data
def load_models():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("rfm_kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    return scaler, kmeans_model

# Load all assets
product_meta = load_metadata()
item_sim_df = download_and_load_similarity()
scaler, kmeans_model = load_models()

# Create ID-to-Name map and update similarity matrix
id_to_name = dict(zip(product_meta['ProductID'], product_meta['ProductName']))
item_sim_df.rename(index=id_to_name, columns=id_to_name, inplace=True)

st.sidebar.header("📍 Choose Module")
mode = st.sidebar.radio("Select Module", ["🔁 Product Recommendation", "🧠 Customer Segmentation"])

# 1️⃣ Product Recommendation Module
if mode == "🔁 Product Recommendation":
    st.subheader("🔁 Recommend Similar Products")

    product_input = st.text_input("Enter Product Name", "white hanging heart t-light holder")
    if st.button("Get Recommendations"):
        if product_input not in item_sim_df.index:
            st.error("❌ Product not found in similarity matrix. Try another name.")
        else:
            top_recommendations = item_sim_df.loc[product_input].sort_values(ascending=False)[1:6]
            st.markdown("### 🔎 Top 5 Similar Products:")
            for i, prod in enumerate(top_recommendations.index, 1):
                st.markdown(f"{i}. **{prod}**")

# 2️⃣ Customer Segmentation Module
elif mode == "🧠 Customer Segmentation":
    st.subheader("🧠 Predict Customer Segment (Using RFM)")

    recency = st.number_input("Recency (in days)", min_value=1, max_value=365, value=30)
    frequency = st.number_input("Frequency (Number of purchases)", min_value=1, max_value=100, value=10)
    monetary = st.number_input("Monetary (Total Spend in USD)", min_value=1, max_value=10000, value=500)

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        input_scaled = scaler.transform(input_df)
        cluster_label = kmeans_model.predict(input_scaled)[0]

        # Mapping Cluster Labels
        segment_map = {
            0: "🟢 High-Value",
            1: "🔵 Regular",
            2: "🟡 Occasional",
            3: "🔴 At-Risk"
        }
        segment = segment_map.get(cluster_label, f"Cluster {cluster_label}")
        st.success(f"✅ Predicted Customer Segment: **{segment}**")
