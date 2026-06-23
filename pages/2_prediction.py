"""
pages/2_prediction.py
Halaman form interaktif prediksi churn nasabah.
User mengisi data nasabah → klik Prediksi → 
model ANN menghasilkan probabilitas dan label churn.
"""

import streamlit as st

from utils import load_artifacts, preprocess_input, predict_churn

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediction",
    page_icon="🔮",
    layout="centered"   # centered lebih cocok untuk form
)

st.title("🔮 Prediksi Customer Churn")
st.write(
    "Isi data nasabah di bawah ini, lalu klik tombol **Prediksi** "
    "untuk melihat apakah nasabah tersebut berpotensi churn atau tidak."
)

# Load artifacts — pakai cache dari utils.py
model, scaler, gender_encoder, feature_columns, comparison_df = load_artifacts()

st.divider()

# ----------------------------------------------------------
# BAGIAN 1: Form Input
# ----------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("📋 Data Nasabah")

    # Bagi form jadi 2 kolom supaya tidak terlalu panjang ke bawah
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            help="Skor kredit nasabah (300–900)"
        )
        country = st.selectbox(
            "Country",
            options=["France", "Germany", "Spain"]
        )
        gender = st.selectbox(
            "Gender",
            options=["Female", "Male"]
        )
        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )
        tenure = st.slider(
            "Tenure (tahun)",
            min_value=0,
            max_value=10,
            value=5,
            help="Sudah berapa tahun menjadi nasabah bank"
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f",
            help="Saldo rekening nasabah"
        )
        products_number = st.selectbox(
            "Jumlah Produk Bank",
            options=[1, 2, 3, 4],
            help="Berapa produk bank yang dimiliki nasabah"
        )
        credit_card = st.radio(
            "Punya Kartu Kredit?",
            options=["Ya", "Tidak"],
            horizontal=True
        )
        active_member = st.radio(
            "Nasabah Aktif?",
            options=["Ya", "Tidak"],
            horizontal=True
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
            format="%.2f",
            help="Estimasi gaji tahunan nasabah"
        )

    # Tombol submit — harus di dalam st.form()
    submitted = st.form_submit_button(
        "🔮 Prediksi Sekarang",
        use_container_width=True
    )

# ----------------------------------------------------------
# BAGIAN 2: Proses Prediksi & Tampilkan Hasil
# ----------------------------------------------------------
# Blok ini hanya dijalankan setelah user klik tombol submit
if submitted:

    # Kumpulkan input dari form ke dalam dict
    # Radio button dikonversi dari "Ya"/"Tidak" ke 1/0
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

    # Preprocessing — urutan sama persis dengan notebook
    X_scaled = preprocess_input(
        raw_input,
        gender_encoder,
        feature_columns,
        scaler
    )

    # Prediksi
    label, proba = predict_churn(model, X_scaled)

    # Tampilkan hasil
    st.divider()
    st.subheader("Hasil Prediksi")

    if label == 1:
        st.error(
            f"⚠️ Nasabah ini diprediksi **BERPOTENSI CHURN**\n\n"
            f"Probabilitas churn: **{proba:.1%}**"
        )
    else:
        st.success(
            f"✅ Nasabah ini diprediksi **AKAN BERTAHAN**\n\n"
            f"Probabilitas churn: **{proba:.1%}**"
        )

    # Progress bar sebagai visualisasi probabilitas
    st.markdown("**Probabilitas Churn:**")
    st.progress(float(proba))
    st.caption(
        f"{proba:.1%} — "
        "Di atas 50% diprediksi churn, di bawah 50% diprediksi bertahan."
    )

    st.divider()

    # Tampilkan detail input yang digunakan untuk prediksi
    with st.expander("📋 Detail data yang diinput"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Credit Score**", credit_score)
            st.write("**Country**", country)
            st.write("**Gender**", gender)
            st.write("**Age**", age)
            st.write("**Tenure**", tenure)
        with col2:
            st.write("**Balance**", f"{balance:,.2f}")
            st.write("**Products Number**", products_number)
            st.write("**Credit Card**", credit_card)
            st.write("**Active Member**", active_member)
            st.write("**Estimated Salary**", f"{estimated_salary:,.2f}")