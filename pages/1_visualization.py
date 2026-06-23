"""
pages/1_visualization.py
Halaman ringkasan dataset dan perbandingan performa model
ANN vs model ML terbaik dari Mini Project 1.
Tidak melakukan training ulang — semua data dibaca dari
file yang sudah di-generate oleh notebook.
"""

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_artifacts

# Konfigurasi halaman
st.set_page_config(
    page_title="Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualization & Model Comparison")

# Load artifacts — pakai cache dari utils.py
# (kalau sudah di-load oleh halaman lain sebelumnya,
# tidak akan load ulang dari disk)
model, scaler, gender_encoder, feature_columns, comparison_df = load_artifacts()

# ----------------------------------------------------------
# BAGIAN 1: Ringkasan Dataset
# ----------------------------------------------------------
st.header("1. Ringkasan Dataset")

DATA_PATH = os.path.join("data", "Bank Customer Churn Prediction.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    # Tampilkan 3 angka kunci di atas sebagai metric card
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Nasabah", f"{df.shape[0]:,}")
    col2.metric("Jumlah Fitur", df.shape[1] - 2)  # kurangi customer_id & churn
    col3.metric("Churn Rate", f"{df['churn'].mean() * 100:.1f}%")

    # Contoh data
    with st.expander("Lihat contoh data (5 baris pertama)"):
        st.dataframe(df.head())

    # Distribusi churn
    st.subheader("Distribusi Target (Churn)")
    col1, col2 = st.columns([1, 2])

    with col1:
        churn_counts = df['churn'].value_counts().rename(
            index={0: 'Bertahan (0)', 1: 'Churn (1)'}
        )
        st.dataframe(churn_counts.rename("Jumlah Nasabah"))

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(
            data=df, x='churn',
            palette=['#2ecc71', '#e74c3c'],
            ax=ax
        )
        ax.set_xticklabels(['Bertahan (0)', 'Churn (1)'])
        ax.set_title('Distribusi Churn')
        ax.set_xlabel('')
        ax.set_ylabel('Jumlah Nasabah')
        st.pyplot(fig)
        plt.close()

else:
    st.warning(
        f"File CSV tidak ditemukan di `{DATA_PATH}`. "
        "Bagian ringkasan dataset dilewati."
    )

st.divider()

# ----------------------------------------------------------
# BAGIAN 2: Perbandingan Model
# ----------------------------------------------------------
st.header("2. Perbandingan Model: ANN vs Model ML Terbaik (Mini Project 1)")

if comparison_df is not None:

    # Tabel perbandingan
    st.subheader("Tabel Metrik")
    st.dataframe(
        comparison_df.style
        .format("{:.4f}")
        .highlight_max(axis=0, color='lightgreen')
        .highlight_min(axis=0, color='#ffe0e0'),
        use_container_width=True
    )
    st.caption(
        "Hijau = nilai tertinggi per metrik | "
        "Merah muda = nilai terendah per metrik"
    )

    # Grafik perbandingan
    st.subheader("Grafik Perbandingan Metrik")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics = [m for m in metrics if m in comparison_df.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_df[metrics].T.plot(
        kind='bar', ax=ax,
        color=['#3498db', '#e67e22'],
        width=0.6
    )
    ax.set_title('Perbandingan Metrik: ANN vs Model ML Terbaik')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(metrics, rotation=0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Interpretasi otomatis berdasarkan angka aktual
    st.subheader("Interpretasi")

    if 'F1-Score' in comparison_df.columns:
        best_model = comparison_df['F1-Score'].idxmax()
        best_f1    = comparison_df['F1-Score'].max()
        worst_f1   = comparison_df['F1-Score'].min()
        selisih    = best_f1 - worst_f1

        st.success(
            f"✅ Model dengan **F1-Score tertinggi**: **{best_model}** "
            f"(F1 = {best_f1:.4f}, selisih {selisih:.4f} dari model lain)"
        )

    if 'ROC-AUC' in comparison_df.columns:
        best_auc = comparison_df['ROC-AUC'].idxmax()
        st.info(
            f"📈 Model dengan **ROC-AUC tertinggi**: **{best_auc}** "
            f"({comparison_df.loc[best_auc, 'ROC-AUC']:.4f}) — "
            "menunjukkan kemampuan lebih baik dalam membedakan "
            "nasabah churn vs tidak churn di semua threshold."
        )

    st.markdown("""
    **Catatan:**
    - Karena data **imbalance** (~80% bertahan, ~20% churn), 
      **F1-Score** dan **Recall** lebih relevan dari Accuracy.
    - **Recall tinggi** = lebih banyak nasabah churn yang berhasil 
      terdeteksi = lebih sedikit yang "lolos" tanpa penanganan.
    - **ROC-AUC** mengukur performa model di semua threshold, 
      bukan hanya di threshold 0.5.
    """)

else:
    st.warning(
        "File `models/comparison_metrics.joblib` belum ditemukan. "
        "Jalankan notebook terlebih dahulu sampai selesai."
    )

st.divider()
st.page_link(
    "pages/2_prediction.py",
    label="➡️ Lanjut ke halaman Prediction",
    icon="🔮"
)