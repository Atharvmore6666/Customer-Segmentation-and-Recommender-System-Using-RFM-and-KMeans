import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="E-Commerce RFM & Recommendation", layout="wide")

# --- DOWNLOAD LARGE FILE FROM GOOGLE DRIVE ---
# Drive file link: https://drive.google.com/file/d/1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n/view?usp=drive_link
FILE_ID = '1Tn94SaJRWTK6_6zNba9wd02EJ4mz8v8n'
ITEM_SIM_PATH = 'item_similarity.pkl'

if not os.path.exists(ITEM_SIM_PATH):
    st.info("Downloading item similarity data...")
    gdown.download(f'https://drive.google.com/uc?id={FILE_ID}', ITEM_SIM_PATH, quiet=False)

# --- LOAD MODELS ---
with open('rfm_kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

item_sim_df = pd.read_pickle(ITEM_SIM_PATH)

# --- HEADER ---
st.title("🛒 E-Commerce Customer Segmentation & Product Recommendation")
st.markdown("""
This app uses **RFM segmentation with KMeans** and **item-based collaborative filtering**  
to classify customers and recommend products they may like.
""")

# --- USER INPUT FOR RFM ---
st.sidebar.header("Enter RFM Metrics")
recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=90)
frequency = st.sidebar.number_input("Frequency (total purchases)", min_value=1, max_value=100, value=5)
monetary = st.sidebar.number_input("Monetary (total spent)", min_value=1, value=500)

rfm_input = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])

# --- SEGMENTATION ---
rfm_scaled = scaler.transform(rfm_input)
cluster = kmeans.predict(rfm_scaled)[0]

st.success(f"🧠 Based on your RFM data, this customer belongs to **Segment {cluster}**.")

# --- PRODUCT RECOMMENDATION ---
st.header("🎯 Product Recommendations")

# Show top N similar items for a sample product
sample_product = st.selectbox("Choose a product you like:", item_sim_df.index.tolist())

def recommend_items(product_name, top_n=5):
    if product_name not in item_sim_df.columns:
        return ["No similar items found."]
    sim_scores = item_sim_df[product_name].sort_values(ascending=False)
    recommendations = sim_scores.iloc[1:top_n+1].index.tolist()
    return recommendations

recommendations = recommend_items(sample_product)

st.subheader(f"Recommended products similar to **{sample_product}**:")
for i, item in enumerate(recommendations, start=1):
    st.write(f"{i}. {item}")

