"""
pages/1_visualization.py
Halaman ringkasan dataset, EDA interaktif, dan perbandingan performa model
ANN vs model ML terbaik dari Mini Project 1.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from utils import load_artifacts

st.set_page_config(
    page_title="Visualization — Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .section-header {
        background: linear-gradient(90deg, #2d3561, #4a5568);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #2d3561;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #374151;
    }
    .winner-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Visualization & Model Comparison")
st.caption("Ringkasan dataset, eksplorasi data interaktif, dan perbandingan performa ANN vs Decision Tree (Tuned).")

model, scaler, gender_encoder, feature_columns, comparison_df = load_artifacts()


# BAGIAN 1: Ringkasan Dataset
st.markdown('<div class="section-header">1. Ringkasan Dataset</div>', unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "Bank Customer Churn Prediction.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nasabah", f"{df.shape[0]:,}")
    col2.metric("Jumlah Fitur", df.shape[1] - 2)
    col3.metric("Churn Rate", f"{df['churn'].mean() * 100:.1f}%")
    col4.metric("Nasabah Churn", f"{df['churn'].sum():,}")

    with st.expander("🔍 Lihat contoh data (10 baris pertama)"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")


    # BAGIAN 2: EDA
    st.markdown('<div class="section-header">2. Eksplorasi Data Interaktif</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Fitur", "🔗 Hubungan dengan Churn", "📋 Statistik Deskriptif"])

    with tab1:
        col_left, col_right = st.columns([1, 3])
        with col_left:
            fitur_pilihan = st.selectbox(
                "Pilih fitur:",
                options=["age", "credit_score", "balance", "estimated_salary",
                         "tenure", "products_number"],
                index=0
            )
            bins = st.slider("Jumlah bins:", min_value=10, max_value=50, value=30)

        with col_right:
            fig, ax = plt.subplots(figsize=(8, 4))
            df_churn    = df[df['churn'] == 1][fitur_pilihan]
            df_no_churn = df[df['churn'] == 0][fitur_pilihan]

            ax.hist(df_no_churn, bins=bins, alpha=0.6, color='#2ecc71',
                    label='Bertahan (0)', edgecolor='white')
            ax.hist(df_churn, bins=bins, alpha=0.7, color='#e74c3c',
                    label='Churn (1)', edgecolor='white')
            ax.set_title(f'Distribusi {fitur_pilihan} berdasarkan Status Churn',
                         fontsize=12, pad=12)
            ax.set_xlabel(fitur_pilihan)
            ax.set_ylabel('Jumlah Nasabah')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Insight secara otomatis
            mean_churn    = df_churn.mean()
            mean_no_churn = df_no_churn.mean()
            diff = mean_churn - mean_no_churn
            arah = "lebih tinggi" if diff > 0 else "lebih rendah"
            st.markdown(f"""
            <div class="insight-box">
            💡 <strong>Insight:</strong> Rata-rata <strong>{fitur_pilihan}</strong> nasabah yang churn
            adalah <strong>{mean_churn:,.1f}</strong>, {arah} {abs(diff):,.1f} poin dibanding
            nasabah yang bertahan (<strong>{mean_no_churn:,.1f}</strong>).
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col_left, col_right = st.columns([1, 3])
        with col_left:
            fitur_kategori = st.selectbox(
                "Pilih fitur kategorikal:",
                options=["country", "gender", "credit_card",
                         "active_member", "products_number"],
                index=0
            )

        with col_right:
            churn_by_feature = df.groupby(fitur_kategori)['churn'].mean().sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#e74c3c' if v == churn_by_feature.max() else '#3498db'
                      for v in churn_by_feature.values]
            bars = ax.bar(churn_by_feature.index.astype(str),
                          churn_by_feature.values * 100,
                          color=colors, edgecolor='white', linewidth=0.5)
            ax.set_title(f'Churn Rate per {fitur_kategori}', fontsize=12, pad=12)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_xlabel(fitur_kategori)
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, churn_by_feature.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Insight secara otomatis
            top_group    = churn_by_feature.idxmax()
            top_rate     = churn_by_feature.max()
            bottom_group = churn_by_feature.idxmin()
            bottom_rate  = churn_by_feature.min()
            st.markdown(f"""
            <div class="insight-box">
            💡 <strong>Insight:</strong> Kelompok <strong>{fitur_kategori} = {top_group}</strong>
            memiliki churn rate tertinggi (<strong>{top_rate:.1%}</strong>), sedangkan
            <strong>{fitur_kategori} = {bottom_group}</strong> memiliki churn rate terendah
            (<strong>{bottom_rate:.1%}</strong>).
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.dataframe(
            df.drop(columns=['customer_id']).describe().round(2),
            use_container_width=True
        )
        st.caption("Statistik deskriptif untuk semua fitur numerik dalam dataset.")

    st.markdown("---")

else:
    st.warning(f"File CSV tidak ditemukan di `{DATA_PATH}`.")


# BAGIAN 3: Perbandingan Model
st.markdown('<div class="section-header">3. Perbandingan Model: ANN vs Decision Tree (Tuned)</div>',
            unsafe_allow_html=True)

if comparison_df is not None:

    tab_tabel, tab_grafik, tab_interpretasi = st.tabs(
        ["📋 Tabel Metrik", "📊 Grafik Perbandingan", "🧠 Interpretasi & Analisis"]
    )

    with tab_tabel:
        st.dataframe(
            comparison_df.style
            .format("{:.4f}")
            .highlight_max(axis=0, color='#d1fae5')
            .highlight_min(axis=0, color='#fee2e2'),
            use_container_width=True
        )
        st.caption("🟢 Hijau = nilai tertinggi per metrik &nbsp;|&nbsp; 🔴 Merah muda = nilai terendah per metrik")

        # Winner per metrik
        st.markdown("**Pemenang per metrik:**")
        cols = st.columns(len(comparison_df.columns))
        for i, metric in enumerate(comparison_df.columns):
            winner = comparison_df[metric].idxmax()
            val    = comparison_df[metric].max()
            with cols[i]:
                st.markdown(f"""
                **{metric}**
                <span class="winner-badge">🏆 {winner.split('(')[0].strip()}</span>
                <br><small>{val:.4f}</small>
                """, unsafe_allow_html=True)

    with tab_grafik:
        metrics = [m for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                   if m in comparison_df.columns]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        x     = np.arange(len(metrics))
        width = 0.35
        models = comparison_df.index.tolist()
        colors = ['#3498db', '#e67e22']

        for i, (model_name, color) in enumerate(zip(models, colors)):
            vals = [comparison_df.loc[model_name, m] for m in metrics]
            bars = axes[0].bar(x + i*width, vals, width, label=model_name,
                               color=color, alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                axes[0].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.005,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

        axes[0].set_title('Perbandingan Metrik (Bar Chart)', fontsize=11, pad=10)
        axes[0].set_xticks(x + width/2)
        axes[0].set_xticklabels(metrics, fontsize=9)
        axes[0].set_ylim(0, 1.08)
        axes[0].set_ylabel('Score')
        axes[0].legend(fontsize=8)
        axes[0].grid(axis='y', alpha=0.3)

        # Radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        axes[1].remove()
        ax_radar = fig.add_subplot(1, 2, 2, polar=True)

        for model_name, color in zip(models, colors):
            vals = [comparison_df.loc[model_name, m] for m in metrics]
            vals += vals[:1]
            ax_radar.plot(angles, vals, 'o-', linewidth=2, label=model_name, color=color)
            ax_radar.fill(angles, vals, alpha=0.15, color=color)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics, fontsize=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Perbandingan Metrik (Radar Chart)', fontsize=11, pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax_radar.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_interpretasi:
        if 'F1-Score' in comparison_df.columns and 'ROC-AUC' in comparison_df.columns:
            ann_f1  = comparison_df.loc["ANN (TensorFlow/Keras)", "F1-Score"] \
                      if "ANN (TensorFlow/Keras)" in comparison_df.index else 0
            ann_auc = comparison_df.loc["ANN (TensorFlow/Keras)", "ROC-AUC"] \
                      if "ANN (TensorFlow/Keras)" in comparison_df.index else 0
            dt_f1   = comparison_df.iloc[1]["F1-Score"] if len(comparison_df) > 1 else 0
            dt_auc  = comparison_df.iloc[1]["ROC-AUC"]  if len(comparison_df) > 1 else 0

            best_model = comparison_df['F1-Score'].idxmax()

            st.success(f"✅ Model dengan **F1-Score tertinggi**: **{best_model}** "
                       f"(F1 = {comparison_df['F1-Score'].max():.4f})")

            st.markdown(f"""
            **Analisis Perbandingan:**

            Berdasarkan tabel perbandingan, **ANN (TensorFlow/Keras) unggul pada metrik
            yang paling relevan** untuk kasus churn prediction — yaitu **F1-Score
            ({ann_f1:.4f} vs {dt_f1:.4f})** dan **ROC-AUC ({ann_auc:.4f} vs {dt_auc:.4f})**.

            **Mengapa F1-Score dan ROC-AUC lebih penting dari Accuracy?**
            Karena dataset ini *imbalance* (~80% bertahan, ~20% churn), model yang selalu
            memprediksi "bertahan" bisa mencapai accuracy 80% tanpa belajar apapun.
            F1-Score dan Recall mengukur seberapa baik model mendeteksi kelas minoritas
            (churn) yang justru paling penting untuk bisnis.

            **Mengapa ANN lebih baik di Recall?**
            ANN dilatih dengan `class_weight='balanced'` yang memberikan penalti lebih
            besar untuk kesalahan prediksi pada kelas churn — konsisten dengan penanganan
            imbalance yang diidentifikasi sejak EDA.

            **Keterbatasan ANN:**
            Meski unggul di Recall dan ROC-AUC, ANN adalah *black box* — sulit dijelaskan
            mengapa seorang nasabah spesifik diprediksi churn. Decision Tree lebih
            interpretatif untuk kebutuhan bisnis.
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.info("🎯 **Gunakan ANN** untuk deteksi dini churn (prioritas recall tinggi)")
            with col2:
                st.info("📋 **Gunakan Decision Tree** untuk interpretasi dan presentasi ke tim bisnis")

else:
    st.warning("File `models/comparison_metrics.joblib` belum ditemukan. "
               "Jalankan notebook terlebih dahulu.")

st.divider()
st.page_link("pages/2_prediction.py",
             label="➡️ Lanjut ke halaman Prediksi", icon="🔮")
