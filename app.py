"""
app.py
Halaman utama (landing page) aplikasi Streamlit.
Jalankan dengan: streamlit run app.py
"""

import streamlit as st

# Konfigurasi halaman — harus di baris pertama sebelum perintah st. apapun
st.set_page_config(
    page_title="Bank Customer Churn",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction")
st.subheader("Mini Project 2 — Artificial Neural Network (TensorFlow/Keras)")

st.markdown("""
Aplikasi ini men-deploy model **Artificial Neural Network (ANN)** yang dilatih
pada *Bank Customer Churn Dataset* untuk memprediksi apakah seorang nasabah
berpotensi **churn** (berhenti menggunakan layanan bank) atau tidak.

Model dibangun di **Mini Project 2** menggunakan TensorFlow/Keras,
dengan dataset dan use case yang sama seperti **Mini Project 1**.
""")

st.divider()

st.markdown("### 📌 Navigasi")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **📊 Visualization**
    
    Ringkasan dataset dan perbandingan performa
    model ANN vs model ML terbaik dari Mini Project 1
    (Decision Tree Tuned).
    """)

with col2:
    st.info("""
    **🔮 Prediction**
    
    Form interaktif untuk memasukkan data nasabah
    dan mendapatkan hasil prediksi churn secara langsung
    dari model ANN.
    """)

st.divider()

st.markdown("""
📂 **Sumber Dataset**: 
[Bank Customer Churn Dataset — Kaggle (Gaurav Topre)]
(https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
""")

st.caption(
    "Gunakan menu navigasi di sidebar kiri untuk berpindah halaman."
)