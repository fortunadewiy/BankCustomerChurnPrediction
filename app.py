"""
app.py
Halaman utama (landing page) aplikasi Streamlit.
Jalankan dengan: streamlit run app.py
"""

import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Bank Churn Prediction — ANN",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------------------------------------
# Custom CSS — tampilan lebih modern dan bersih
# ----------------------------------------------------------
st.markdown("""
<style>
    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 50%, #1a1f3a 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #3d4a8a;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .main-header p {
        color: #a8b4d8;
        font-size: 1rem;
        margin: 0;
    }

    /* Metric card */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e8ecf4;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3561;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    .metric-card .delta {
        font-size: 0.8rem;
        color: #10b981;
        margin-top: 0.3rem;
    }

    /* Nav card */
    .nav-card {
        background: #f8faff;
        border: 1px solid #dde3f4;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: border-color 0.2s;
    }
    .nav-card:hover {
        border-color: #2d3561;
    }
    .nav-card .icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }
    .nav-card h3 {
        color: #1a1f3a;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .nav-card p {
        color: #6b7280;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-blue  { background: #dbeafe; color: #1e40af; }
    .badge-green { background: #d1fae5; color: #065f46; }
    .badge-purple{ background: #ede9fe; color: #5b21b6; }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.82rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Customer Churn Prediction</h1>
    <p>Mini Project 2 — Artificial Neural Network (TensorFlow/Keras) &nbsp;|&nbsp; Deployment dengan Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# METRIC CARDS — load dari comparison_metrics.joblib
# ----------------------------------------------------------
MODEL_DIR = "models"
comparison_path = os.path.join(MODEL_DIR, "comparison_metrics.joblib")

if os.path.exists(comparison_path):
    comparison_df = joblib.load(comparison_path)

    # Ambil metrik ANN
    if "ANN (TensorFlow/Keras)" in comparison_df.index:
        ann = comparison_df.loc["ANN (TensorFlow/Keras)"]
        acc    = ann.get("Accuracy", 0)
        f1     = ann.get("F1-Score", 0)
        recall = ann.get("Recall", 0)
        auc    = ann.get("ROC-AUC", 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{acc:.1%}</div>
                <div class="label">Accuracy (ANN)</div>
                <div class="delta">↑ Model terbaik</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{f1:.1%}</div>
                <div class="label">F1-Score Churn</div>
                <div class="delta">↑ Metrik utama</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{recall:.1%}</div>
                <div class="label">Recall Churn</div>
                <div class="delta">↑ Deteksi churn</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{auc:.3f}</div>
                <div class="label">ROC-AUC</div>
                <div class="delta">↑ Diskriminasi model</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------
# DESKRIPSI SINGKAT
# ----------------------------------------------------------
st.markdown("""
Aplikasi ini men-deploy model **Artificial Neural Network (ANN)** yang dilatih pada
*Bank Customer Churn Dataset* untuk memprediksi apakah seorang nasabah berpotensi
**churn** (berhenti menggunakan layanan bank) atau tidak.
""")

# Tech stack badges
st.markdown("""
<span class="badge badge-blue">TensorFlow/Keras</span>
<span class="badge badge-blue">Scikit-learn</span>
<span class="badge badge-green">Streamlit</span>
<span class="badge badge-green">Python 3.10</span>
<span class="badge badge-purple">ANN · Dropout · Early Stopping</span>
<span class="badge badge-purple">GridSearchCV</span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ----------------------------------------------------------
# NAVIGASI
# ----------------------------------------------------------
st.markdown("### 🗺️ Navigasi Halaman")
st.caption("Gunakan sidebar kiri atau klik deskripsi di bawah untuk berpindah halaman.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">📊</div>
        <h3>Visualization & Model Comparison</h3>
        <p>
            Ringkasan statistik dataset, distribusi churn,
            dan perbandingan performa model <strong>ANN</strong> vs
            <strong>Decision Tree (Tuned)</strong> dari Mini Project 1
            menggunakan metrik Accuracy, F1-Score, Recall, dan ROC-AUC.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">🔮</div>
        <h3>Prediksi Customer Churn</h3>
        <p>
            Form interaktif untuk memasukkan data nasabah baru
            dan mendapatkan hasil prediksi churn secara langsung
            dari model ANN, lengkap dengan probabilitas, profil risiko,
            dan rekomendasi aksi bisnis.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ----------------------------------------------------------
# TENTANG PROJECT
# ----------------------------------------------------------
with st.expander("📖 Tentang Project ini"):
    st.markdown("""
    **Mini Project 2** merupakan lanjutan dari Mini Project 1 yang menggunakan
    dataset dan use case yang sama, namun membangun model menggunakan
    **Artificial Neural Network (ANN)** berbasis TensorFlow/Keras.

    **Keputusan Arsitektur ANN:**
    - **2 Hidden Layer** (16 → 8 neuron): cukup untuk menangkap pola non-linear
      pada data tabular 11 fitur tanpa overfitting
    - **Aktivasi ReLU**: efisien, menghindari vanishing gradient
    - **Dropout** (0.3 → 0.2): regularisasi untuk mencegah overfitting
    - **Early Stopping** (patience=5): menghentikan training otomatis saat
      validation loss tidak membaik
    - **class_weight='balanced'**: menangani imbalance data (~80% bertahan,
      ~20% churn) di level algoritma

    **Perbandingan dengan Mini Project 1:**
    Model ANN dibandingkan secara eksplisit dengan Decision Tree (Tuned)
    sebagai model ML terbaik dari Mini Project 1, menggunakan metrik
    F1-Score dan ROC-AUC sebagai metrik utama karena data imbalance.
    """)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("""
<div class="footer">
    📂 Dataset: <a href="https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset" target="_blank">
    Bank Customer Churn Dataset — Kaggle (Gaurav Topre)</a>
    &nbsp;·&nbsp; Mini Project 2 &nbsp;·&nbsp; TensorFlow/Keras + Streamlit
</div>
""", unsafe_allow_html=True)
