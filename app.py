import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

model = joblib.load("model/model_diabetes.pkl")
scaler = joblib.load("model/scaler_diabetes.pkl")

feature_names = [
    "Jumlah_Kehamilan", "Glukosa", "Tekanan_Darah", "Ketebalan_Kulit",
    "Insulin", "BMI", "Riwayat_Keluarga", "Usia"
]

st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("📊 Prediksi Diabetes Menggunakan Regresi Logistik")

opsi_menu = {
    "🧪 Prediksi Data Baru": "prediksi",
    "📊 Model Visualisasi": "visualisasi",
    "📁 Riwayat Prediksi": "riwayat",
    "📈 Ringkasan Statistik": "ringkasan"
}
menu = st.sidebar.radio("📂 Menu Navigasi", list(opsi_menu.keys()))
selected = opsi_menu[menu]

# ========== 🧪 PREDIKSI ==========
if selected == "prediksi":
    st.header("🧪 Prediksi Data Baru")
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            kehamilan = st.number_input("Jumlah Kehamilan", 0, 20, 1)
            glukosa = st.number_input("Kadar Glukosa (mg/dL)", 0, 300, 120)
            tekanan = st.number_input("Tekanan Darah (mm Hg)", 0, 200, 70)
            kulit = st.number_input("Ketebalan Kulit Triceps (mm)", 0, 100, 20)
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 79)
            bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 28.5)
            riwayat = st.number_input("Riwayat Diabetes Keluarga", 0.0, 2.5, 0.5)
            usia = st.number_input("Usia (tahun)", 1, 120, 30)
        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        data_input = np.array([[kehamilan, glukosa, tekanan, kulit, insulin, bmi, riwayat, usia]])
        scaled_input = scaler.transform(data_input)
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        hasil = "🟥 Positif Diabetes" if pred == 1 else "🟩 Negatif Diabetes"

        st.subheader("Hasil Prediksi:")
        st.markdown(f"**Status:** {hasil}")
        st.markdown(f"**Probabilitas:** {prob:.2%}")

        # Interpretasi
        koef = model.coef_[0]
        dampak = [(feature_names[i], scaled_input[0][i] * koef[i]) for i in range(len(feature_names))]
        dampak = sorted(dampak, key=lambda x: abs(x[1]), reverse=True)

        st.subheader("Fitur Paling Berpengaruh:")
        for i, (fitur, skor) in enumerate(dampak[:3]):
            arah = "meningkatkan" if skor > 0 else "menurunkan"
            st.write(f"{i+1}. {fitur} — {arah} risiko diabetes (kontribusi: {skor:.4f})")

        hasil_df = pd.DataFrame([{
            "Jumlah_Kehamilan": kehamilan,
            "Glukosa": glukosa,
            "Tekanan_Darah": tekanan,
            "Ketebalan_Kulit": kulit,
            "Insulin": insulin,
            "BMI": bmi,
            "Riwayat_Keluarga": riwayat,
            "Usia": usia,
            "Prediksi": hasil,
            "Probabilitas": prob,
            "Faktor_Terkuat_1": dampak[0][0],
            "Faktor_Terkuat_2": dampak[1][0],
            "Faktor_Terkuat_3": dampak[2][0]
        }])

        os.makedirs("hasil", exist_ok=True)
        file_path = "hasil/hasil_prediksi.csv"
        if os.path.exists(file_path):
            hasil_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            hasil_df.to_csv(file_path, index=False)
        st.success("✅ Hasil disimpan ke 'hasil/hasil_prediksi.csv'")

# ========== 📁 RIWAYAT ==========
elif selected == "riwayat":
    st.header("📁 Riwayat Prediksi")

    st.markdown("### 🔧 Perbaiki File CSV (jika ada error)")
    if st.button("🛠 Perbaiki Struktur File hasil_prediksi.csv"):
        try:
            df_fix = pd.read_csv("hasil/hasil_prediksi.csv", on_bad_lines='skip')

            kolom_dibuat = []
            for kolom in ["Faktor_Terkuat_1", "Faktor_Terkuat_2", "Faktor_Terkuat_3"]:
                if kolom not in df_fix.columns:
                    df_fix[kolom] = "-"
                    kolom_dibuat.append(kolom)

            if kolom_dibuat:
                df_fix.to_csv("hasil/hasil_prediksi.csv", index=False)
                st.success(f"✅ Berhasil menambahkan kolom: {', '.join(kolom_dibuat)}")
            else:
                st.info("✔️ Semua kolom sudah lengkap. Tidak ada yang diubah.")
        except Exception as e:
            st.error(f"❌ Gagal memperbaiki file: {e}")

    # Tampilkan riwayat prediksi terbaru
    try:
        df_history = pd.read_csv("hasil/hasil_prediksi.csv", on_bad_lines='skip')
        st.markdown("### 📄 Data Prediksi Terbaru:")
        st.dataframe(df_history.tail(10).reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Gagal menampilkan riwayat: {e}")


# ========== 📊 VISUALISASI ==========
elif selected == "visualisasi":
    st.header("📊 Model Visualisasi")
    visual_dir = "visualisasi"

    gambar = {
        "Matriks Konfusi": "confusion_matrix.png",
        "ROC Curve": "roc_curve.png",
        "Pentingnya Fitur": "feature_importance.png",
        "Distribusi Kelas": "distribusi_kelas.png",
        "Korelasi Fitur": "korelasi.png"
    }

    for judul, nama_file in gambar.items():
        path = os.path.join(visual_dir, nama_file)
        if os.path.exists(path):
            st.subheader(judul)
            st.image(path, use_container_width=True)
        else:
            st.warning(f"Gambar `{nama_file}` tidak ditemukan.")


# ========== 📈 RINGKASAN ==========
elif selected == "ringkasan":
    st.header("📈 Ringkasan statistik prediksi")
    try:
        df = pd.read_csv("hasil/hasil_prediksi.csv", on_bad_lines='skip')
        total = len(df)
        positif = df["Prediksi"].str.contains("Positif").sum()
        negatif = total - positif

        st.write(f"Jumlah Prediksi: **{total}**")
        st.write(f"Positif Diabetes: **{positif} ({(positif/total)*100:.1f}%)**")
        st.write(f"Negatif Diabetes: **{negatif} ({(negatif/total)*100:.1f}%)**")

        # Adaptif: cari kombinasi kolom rata-rata
        if {"Glukosa", "BMI", "Usia"}.issubset(df.columns):
            rata2 = df[["Glukosa", "BMI", "Usia"]].mean()
        elif {"Glucose", "BMI", "Age"}.issubset(df.columns):
            rata2 = df[["Glucose", "BMI", "Age"]].mean()
        else:
            st.warning("⚠️ Kolom rata-rata tidak ditemukan.")
            rata2 = None

        st.markdown("### 📊 Data Rata-rata:")
        if rata2 is not None:
            for kolom, nilai in rata2.items():
                st.write(f"- {kolom}: {nilai:.2f}")

        # Cek apakah kolom faktor dominan ada
        faktor_cols = ["Faktor_Terkuat_1", "Faktor_Terkuat_2", "Faktor_Terkuat_3"]
        faktor_cols = [col for col in faktor_cols if col in df.columns]

        if faktor_cols:
            semua_faktor = pd.concat([df[col] for col in faktor_cols])
            frekuensi = semua_faktor.value_counts().head(3)
            st.markdown("### ⭐ Fitur Dominan Terbanyak:")
            for i, (fitur, jumlah) in enumerate(frekuensi.items(), 1):
                st.write(f"{i}. {fitur} — muncul {jumlah}x")
        else:
            st.warning("⚠️ Kolom fitur dominan tidak ditemukan.")

    except Exception as e:
        st.error(f"Gagal menghitung ringkasan: {e}")
