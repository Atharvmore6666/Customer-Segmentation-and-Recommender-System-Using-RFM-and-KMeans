import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

st.set_page_config(page_title="Customer Segmentation & Product Recommender", layout="wide")
st.title("🛒 Customer Segmentation & Product Recommendation App")

# ---------------------------
# Load Pickle Models
# ---------------------------
with open("rfm_kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Download item_similarity.pkl from Google Drive
# ---------------------------
@st.cache_data
def download_and_load_similarity():
    file_id = "1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "item_similarity.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return pd.read_pickle(output)

item_sim_df = download_and_load_similarity()
product_names = item_sim_df.index.tolist()

# ---------------------------
# Sidebar Options
# ---------------------------
option = st.sidebar.radio("Select Option", ["🔁 RFM Segmentation", "🛍️ Product Recommendation"])

# ---------------------------
# RFM Prediction
# ---------------------------
if option == "🔁 RFM Segmentation":
    st.header("📊 Predict Customer Segment using RFM values")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, max_value=1000, value=30)
    with col2:
        frequency = st.number_input("Frequency (# of orders)", min_value=0, max_value=100, value=5)
    with col3:
        monetary = st.number_input("Monetary (spend)", min_value=0, max_value=10000, value=500)

    if st.button("Predict Segment"):
        rfm_input = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_input)
        segment = kmeans.predict(rfm_scaled)[0]
        st.success(f"🧠 Predicted Segment: **Cluster {segment}**")

# ---------------------------
# Product Recommendation
# ---------------------------
elif option == "🛍️ Product Recommendation":
    st.header("🔎 Get Product Recommendations")

    selected_product = st.selectbox("Choose a product", options=product_names)

    if st.button("Recommend"):
        if selected_product in item_sim_df.index:
            similar_products = item_sim_df[selected_product].sort_values(ascending=False).iloc[1:6]
            st.subheader("🛍️ Top 5 Similar Products:")
            for i, (product, score) in enumerate(similar_products.items(), 1):
                st.write(f"**{i}. {product}** (Similarity: {score:.2f})")
        else:
            st.error("Product not found in similarity matrix.")
