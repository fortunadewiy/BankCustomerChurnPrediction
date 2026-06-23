import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import keras


MODEL_DIR = "models"


@st.cache_resource
def load_artifacts():
    """
    Load model ANN dan semua artifacts preprocessing dari folder models/.
    
    @st.cache_resource memastikan fungsi ini hanya dijalankan SEKALI
    selama app berjalan — hasilnya disimpan di memory dan dipakai ulang
    setiap kali halaman diakses, tanpa load ulang dari disk.
    """
    # Definisikan path semua file yang dibutuhkan
    paths = {
        "model"          : os.path.join(MODEL_DIR, "ann_churn_model.keras"),
        "scaler"         : os.path.join(MODEL_DIR, "scaler.joblib"),
        "gender_encoder" : os.path.join(MODEL_DIR, "gender_encoder.joblib"),
        "feature_columns": os.path.join(MODEL_DIR, "feature_columns.joblib"),
    }

    # Cek semua file wajib ada sebelum load
    # Kalau ada yang hilang, tampilkan pesan error yang jelas ke user
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        st.error(
            "File berikut tidak ditemukan di folder `models/`:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nJalankan notebook `mini_project_2_ann_churn.ipynb` "
              "dari awal sampai selesai terlebih dahulu."
        )
        st.stop()

    # Load semua artifacts
    model           = keras.models.load_model(paths["model"])
    scaler          = joblib.load(paths["scaler"])
    gender_encoder  = joblib.load(paths["gender_encoder"])
    feature_columns = joblib.load(paths["feature_columns"])

    # comparison_metrics opsional — halaman visualization butuh ini
    # tapi halaman prediction tidak, jadi tidak di-stop kalau tidak ada
    comparison_path = os.path.join(MODEL_DIR, "comparison_metrics.joblib")
    comparison_df   = (joblib.load(comparison_path)
                       if os.path.exists(comparison_path) else None)

    return model, scaler, gender_encoder, feature_columns, comparison_df


def preprocess_input(raw_input, gender_encoder, feature_columns, scaler):
    """
    Ubah input mentah dari form Streamlit menjadi array siap prediksi.

    Urutan langkah HARUS sama persis dengan preprocessing di notebook:
    1. Label Encoding gender
    2. One-Hot Encoding country (drop_first=True → France jadi baseline)
    3. Tambahkan kolom yang hilang dengan nilai 0
    4. Urutkan kolom sesuai feature_columns saat training
    5. StandardScaler transform

    raw_input contoh:
    {
        'credit_score'    : 650,
        'country'         : 'France',
        'gender'          : 'Female',
        'age'             : 35,
        'tenure'          : 5,
        'balance'         : 50000.0,
        'products_number' : 2,
        'credit_card'     : 1,
        'active_member'   : 1,
        'estimated_salary': 60000.0,
    }
    """
    df = pd.DataFrame([raw_input])

    # Step 1: Label Encoding gender — sama persis dengan notebook
    df['gender'] = gender_encoder.transform(df['gender'])

    # Step 2: One-Hot Encoding country — drop_first=True seperti di notebook
    # France jadi baseline (tidak muncul sebagai kolom tersendiri)
    df = pd.get_dummies(df, columns=['country'])

    # Step 3: Tambahkan kolom dummy yang tidak muncul dari input ini
    # Contoh: kalau user pilih France, kolom country_Germany
    # dan country_Spain tidak akan terbentuk dari get_dummies
    # → perlu ditambahkan manual dengan nilai 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Step 4: Urutkan kolom PERSIS sama dengan urutan saat training
    # Ini krusial — model ANN sensitif terhadap urutan kolom
    df = df[feature_columns]

    # Step 5: Scaling — pakai scaler yang sudah di-fit ke data training
    X_scaled = scaler.transform(df)

    return X_scaled


def predict_churn(model, X_scaled):
    """
    Jalankan prediksi dan kembalikan label dan probabilitas.

    Return:
        label : int  — 1 = churn, 0 = tidak churn
        proba : float — probabilitas churn (0.0 – 1.0)
    """
    # model.predict() mengembalikan array 2D shape (1, 1)
    # .ravel()[0] mengubahnya jadi satu angka float
    proba = model.predict(X_scaled, verbose=0).ravel()[0]
    label = int(proba >= 0.5)

    return label, float(proba)