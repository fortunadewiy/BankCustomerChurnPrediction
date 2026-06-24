"""
pages/2_prediction.py
Halaman form interaktif prediksi churn nasabah.
Dilengkapi dengan risk profile, faktor risiko, dan rekomendasi aksi bisnis.
"""

import streamlit as st
import pandas as pd

from utils import load_artifacts, preprocess_input, predict_churn

st.set_page_config(
    page_title="Prediksi Churn — ANN",
    page_icon="🔮",
    layout="wide"
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------
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
    .result-churn {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #f0fdf4, #d1fae5);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-churn h2 { color: #b91c1c; margin: 0 0 0.5rem 0; font-size: 1.6rem; }
    .result-safe  h2 { color: #065f46; margin: 0 0 0.5rem 0; font-size: 1.6rem; }
    .result-churn p  { color: #7f1d1d; margin: 0; font-size: 1rem; }
    .result-safe  p  { color: #064e3b; margin: 0; font-size: 1rem; }

    .risk-factor {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        color: #431407;
    }
    .safe-factor {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        color: #064e3b;
    }
    .rekomendasi-card {
        background: #f8faff;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .rekomendasi-card h4 { color: #3730a3; margin: 0 0 0.3rem 0; font-size: 0.95rem; }
    .rekomendasi-card p  { color: #4b5563; margin: 0; font-size: 0.85rem; line-height: 1.5; }

    .proba-gauge {
        text-align: center;
        padding: 1rem;
    }
    .proba-number {
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }
    .proba-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Prediksi Customer Churn")
st.caption("Masukkan data nasabah, lalu klik Prediksi untuk mendapatkan hasil dari model ANN beserta analisis risiko.")

model, scaler, gender_encoder, feature_columns, comparison_df = load_artifacts()

# ----------------------------------------------------------
# FORM INPUT
# ----------------------------------------------------------
st.markdown('<div class="section-header">Data Nasabah</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Profil Nasabah**")
        gender = st.selectbox("Gender", options=["Female", "Male"])
        age    = st.slider("Usia (tahun)", min_value=18, max_value=100, value=35)
        country = st.selectbox("Negara", options=["France", "Germany", "Spain"])

    with col2:
        st.markdown("**💰 Informasi Keuangan**")
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650,
                                 help="Skor kredit nasabah (300–900)")
        balance = st.number_input("Saldo Rekening", min_value=0.0, value=50000.0,
                                  step=1000.0, format="%.2f")
        estimated_salary = st.number_input("Estimasi Gaji Tahunan", min_value=0.0,
                                           value=60000.0, step=1000.0, format="%.2f")

    with col3:
        st.markdown("**🏦 Hubungan dengan Bank**")
        tenure = st.slider("Lama Menjadi Nasabah (tahun)", min_value=0, max_value=10, value=5)
        products_number = st.selectbox("Jumlah Produk Bank", options=[1, 2, 3, 4],
                                       help="Berapa produk bank yang dimiliki")
        credit_card   = st.radio("Memiliki Kartu Kredit?", options=["Ya", "Tidak"], horizontal=True)
        active_member = st.radio("Nasabah Aktif?",         options=["Ya", "Tidak"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

# ----------------------------------------------------------
# HASIL PREDIKSI
# ----------------------------------------------------------
if submitted:
    raw_input = {
        'credit_score'    : credit_score,
        'country'         : country,
        'gender'          : gender,
        'age'             : age,
        'tenure'          : tenure,
        'balance'         : balance,
        'products_number' : products_number,
        'credit_card'     : 1 if credit_card == "Ya" else 0,
        'active_member'   : 1 if active_member == "Ya" else 0,
        'estimated_salary': estimated_salary,
    }

    X_scaled     = preprocess_input(raw_input, gender_encoder, feature_columns, scaler)
    label, proba = predict_churn(model, X_scaled)

    st.markdown("---")
    st.markdown('<div class="section-header">Hasil Prediksi</div>', unsafe_allow_html=True)

    # Hasil utama + probabilitas
    col_result, col_proba = st.columns([2, 1])

    with col_result:
        if label == 1:
            st.markdown(f"""
            <div class="result-churn">
                <h2>⚠️ BERPOTENSI CHURN</h2>
                <p>Model ANN memprediksi nasabah ini berisiko meninggalkan layanan bank.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h2>✅ DIPREDIKSI BERTAHAN</h2>
                <p>Model ANN memprediksi nasabah ini akan tetap menggunakan layanan bank.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_proba:
        proba_color = "#ef4444" if proba >= 0.5 else "#10b981"
        st.markdown(f"""
        <div class="proba-gauge">
            <div class="proba-number" style="color: {proba_color};">{proba:.1%}</div>
            <div class="proba-label">Probabilitas Churn</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(proba))
        st.caption("0% = pasti bertahan · 100% = pasti churn · Threshold: 50%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------------
    # RISK FACTORS & PROFILE
    # ----------------------------------------------------------
    col_risk, col_rekom = st.columns([1, 1])

    with col_risk:
        st.markdown("**🔍 Analisis Faktor Risiko**")

        # Faktor risiko berdasarkan temuan EDA dari MP1
        risk_factors  = []
        safe_factors  = []

        if age > 45:
            risk_factors.append(f"🔴 Usia {age} tahun — nasabah senior cenderung lebih sering churn")
        else:
            safe_factors.append(f"🟢 Usia {age} tahun — dalam rentang usia yang relatif aman")

        if products_number == 1:
            risk_factors.append("🔴 Hanya memiliki 1 produk bank — ketergantungan rendah")
        elif products_number >= 3:
            risk_factors.append("🔴 Memiliki 3+ produk — pola ini sering berkorelasi dengan churn")
        else:
            safe_factors.append(f"🟢 Memiliki {products_number} produk bank — jumlah yang stabil")

        if active_member == "Tidak":
            risk_factors.append("🔴 Nasabah tidak aktif — risiko churn meningkat signifikan")
        else:
            safe_factors.append("🟢 Nasabah aktif — keterlibatan dengan bank masih baik")

        if country == "Germany":
            risk_factors.append("🔴 Negara: Jerman — churn rate di Jerman lebih tinggi dari rata-rata")
        else:
            safe_factors.append(f"🟢 Negara: {country} — churn rate relatif lebih rendah")

        if balance > 100000:
            risk_factors.append(f"🔴 Saldo tinggi (Rp {balance:,.0f}) — nasabah saldo besar lebih aktif membandingkan bank lain")
        elif balance == 0:
            risk_factors.append("🔴 Saldo nol — tidak ada aset yang menahan nasabah di bank ini")
        else:
            safe_factors.append(f"🟢 Saldo {balance:,.0f} — dalam rentang yang normal")

        if credit_score < 500:
            risk_factors.append(f"🔴 Credit score rendah ({credit_score}) — profil kredit berisiko")
        elif credit_score >= 700:
            safe_factors.append(f"🟢 Credit score baik ({credit_score}) — profil kredit sehat")

        # Tampilkan faktor
        if risk_factors:
            st.markdown("**Faktor yang meningkatkan risiko churn:**")
            for f in risk_factors:
                st.markdown(f'<div class="risk-factor">{f}</div>', unsafe_allow_html=True)

        if safe_factors:
            st.markdown("**Faktor yang menurunkan risiko churn:**")
            for f in safe_factors:
                st.markdown(f'<div class="safe-factor">{f}</div>', unsafe_allow_html=True)

    with col_rekom:
        st.markdown("**💼 Rekomendasi Aksi Bisnis**")

        if label == 1:
            # Rekomendasi untuk nasabah berisiko churn
            rekoms = []

            if active_member == "Tidak":
                rekoms.append({
                    "judul": "🎯 Program Re-engagement",
                    "isi": "Hubungi nasabah dengan penawaran personal: cashback, diskon biaya administrasi, atau undangan ke program loyalitas untuk meningkatkan keterlibatan."
                })
            if products_number <= 1:
                rekoms.append({
                    "judul": "📦 Cross-selling Produk",
                    "isi": "Tawarkan produk tambahan yang sesuai profil (tabungan berjangka, kartu kredit, atau asuransi) untuk meningkatkan ketergantungan nasabah pada bank."
                })
            if balance > 100000:
                rekoms.append({
                    "judul": "💎 Layanan Prioritas",
                    "isi": "Nasabah dengan saldo tinggi perlu penanganan khusus. Assign relationship manager dan tawarkan layanan premium atau suku bunga kompetitif."
                })
            if country == "Germany":
                rekoms.append({
                    "judul": "🇩🇪 Segmen Jerman — Perhatian Khusus",
                    "isi": "Nasabah di Jerman memiliki churn rate lebih tinggi. Lakukan survei kepuasan dan identifikasi kebutuhan spesifik segmen ini."
                })
            if age > 45:
                rekoms.append({
                    "judul": "👴 Program Nasabah Senior",
                    "isi": "Sediakan layanan yang lebih personal dan mudah diakses untuk nasabah senior, termasuk layanan antar jemput dokumen atau konsultasi tatap muka."
                })

            # Default kalau tidak ada kondisi spesifik
            if not rekoms:
                rekoms.append({
                    "judul": "📞 Proactive Outreach",
                    "isi": "Hubungi nasabah secara proaktif untuk memahami kebutuhan dan kekhawatiran mereka sebelum memutuskan untuk pindah ke bank lain."
                })

            for r in rekoms[:3]:  # maksimal 3 rekomendasi
                st.markdown(f"""
                <div class="rekomendasi-card">
                    <h4>{r['judul']}</h4>
                    <p>{r['isi']}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Rekomendasi untuk nasabah yang diprediksi bertahan
            st.markdown(f"""
            <div class="rekomendasi-card">
                <h4>✅ Pertahankan Kepuasan</h4>
                <p>Nasabah ini diprediksi bertahan. Tetap jaga kepuasan dengan pelayanan yang konsisten dan komunikasi berkala.</p>
            </div>
            <div class="rekomendasi-card">
                <h4>📈 Upselling Opportunity</h4>
                <p>Nasabah loyal adalah kandidat ideal untuk penawaran produk premium atau peningkatan layanan yang menguntungkan kedua pihak.</p>
            </div>
            <div class="rekomendasi-card">
                <h4>🌟 Program Loyalitas</h4>
                <p>Daftarkan nasabah ke program loyalitas untuk memperkuat hubungan jangka panjang dan mengurangi risiko churn di masa depan.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ----------------------------------------------------------
    # DETAIL INPUT
    # ----------------------------------------------------------
    with st.expander("📋 Detail lengkap data yang diinput"):
        detail_df = pd.DataFrame({
            "Fitur": ["Gender", "Usia", "Negara", "Credit Score", "Saldo",
                      "Estimasi Gaji", "Lama Nasabah", "Jumlah Produk",
                      "Kartu Kredit", "Nasabah Aktif"],
            "Nilai": [gender, age, country, credit_score,
                      f"{balance:,.2f}", f"{estimated_salary:,.2f}",
                      f"{tenure} tahun", products_number,
                      credit_card, active_member]
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.caption(
        "⚠️ Hasil prediksi ini dihasilkan oleh model machine learning dan bersifat probabilistik. "
        "Keputusan bisnis akhir tetap harus mempertimbangkan konteks dan judgement manusia."
    )
