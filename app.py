import streamlit as st
import pandas as pd
import gdown
import os
import pickle

st.set_page_config(page_title="🛍️ Product Recommender & Customer Segmentation", layout="centered")

# ------------------- Load Assets ------------------- #
@st.cache_data

def load_product_metadata():
    return pd.read_csv("product_metadata.csv")

@st.cache_data

def download_and_load_similarity():
    url = "https://drive.google.com/uc?id=1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n"
    output = "item_similarity.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return pd.read_pickle(output)

@st.cache_data

def load_cluster_model():
    with open("scaler.pkl", "rb") as f1:
        scaler = pickle.load(f1)
    with open("rfm_kmeans_model.pkl", "rb") as f2:
        kmeans = pickle.load(f2)
    return scaler, kmeans

# ------------------- App UI ------------------- #
st.title("🛍️ Product Recommendation & Customer Segmentation")
st.markdown("---")

# Load all required data
metadata_df = load_product_metadata()
sim_df = download_and_load_similarity()
scaler, kmeans = load_cluster_model()

# Define product name column
product_col = "Description"  # Change this if needed

# 1️⃣ Product Recommendation Module
st.header("🎯 1. Product Recommendation")

product_input = st.text_input("Enter Product Name:")

if st.button("Get Recommendations"):
    if product_input not in sim_df.columns:
        st.error("Product not found in similarity matrix. Please check the name.")
    else:
        st.subheader(f"🧠 Top 5 Similar Products to: `{product_input}`")
        recommendations = sim_df[product_input].sort_values(ascending=False)[1:6].index.tolist()
        for i, rec in enumerate(recommendations, start=1):
            st.markdown(f"**{i}.** {rec}")

st.markdown("---")

# 2️⃣ Customer Segmentation Module
st.header("🎯 2. Customer Segmentation")

col1, col2, col3 = st.columns(3)
with col1:
    recency = st.number_input("Recency (days)", min_value=0, value=30)
with col2:
    frequency = st.number_input("Frequency (purchases)", min_value=0, value=5)
with col3:
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=200.0)

if st.button("Predict Cluster"):
    input_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster_label = kmeans.predict(input_scaled)[0]

    cluster_names = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }
    label = cluster_names.get(cluster_label, f"Cluster {cluster_label}")
    st.success(f"🎯 Predicted Customer Segment: **{label}**")
