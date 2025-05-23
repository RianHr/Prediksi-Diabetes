# src/prediksi.py

import numpy as np
import pandas as pd
import os

def prediksi_data_baru(model, scaler, feature_names):
    print("\n=== PREDIKSI DIABETES UNTUK DATA BARU ===")

    try:
        # Input data dari pengguna
        data_input = []
        input_labels = [
            "Jumlah Kehamilan",
            "Kadar Glukosa (mg/dL)",
            "Tekanan Darah (mm Hg)",
            "Ketebalan Kulit Triceps (mm)",
            "Insulin (mu U/ml)",
            "BMI (kg/m²)",
            "Fungsi Riwayat Diabetes Keluarga (contoh: 0.543)",
            "Usia (tahun)"
        ]

        for label in input_labels:
            while True:
                try:
                    nilai = float(input(f"{label}: "))
                    data_input.append(nilai)
                    break
                except ValueError:
                    print("⚠️ Masukkan harus berupa angka. Silakan coba lagi.")

        # Ubah ke array dan skalakan
        input_data = np.array([data_input])
        input_scaled = scaler.transform(input_data)

        # Prediksi
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        hasil = "Positif Diabetes" if pred == 1 else "Negatif Diabetes"
        print(f"\n=== HASIL PREDIKSI ===")
        print(f"Status: {hasil}")
        print(f"Probabilitas Diabetes: {prob:.2%}")

        # Interpretasi
        print("\n=== INTERPRETASI ===")
        if pred == 1:
            if prob > 0.75:
                print("Risiko diabetes sangat tinggi. Konsultasikan dengan dokter segera.")
            else:
                print("Risiko diabetes terdeteksi. Disarankan konsultasi medis.")
        else:
            if prob > 0.4:
                print("Hasil negatif, namun risiko cukup dekat dengan ambang batas.")
                print("Tetap jaga pola hidup sehat.")
            else:
                print("Risiko rendah. Terus pertahankan gaya hidup sehat.")

        # Top 3 faktor berpengaruh
        print("\n=== FAKTOR TERPENTING ===")
        koef = model.coef_[0]
        dampak = [(feature_names[i], input_scaled[0][i] * koef[i]) for i in range(len(feature_names))]
        dampak = sorted(dampak, key=lambda x: abs(x[1]), reverse=True)

        top_features = []
        for i, (fitur, skor) in enumerate(dampak[:3]):
            arah = "meningkatkan" if skor > 0 else "menurunkan"
            print(f"{i+1}. {fitur}: {arah} risiko diabetes (kontribusi: {skor:.4f})")
            top_features.append(fitur)

        # Simpan hasil
        hasil_df = pd.DataFrame([{
            "Jumlah_Kehamilan": data_input[0],
            "Kadar_Glukosa": data_input[1],
            "Tekanan_Darah": data_input[2],
            "Ketebalan_Kulit": data_input[3],
            "Insulin": data_input[4],
            "BMI": data_input[5],
            "Riwayat_Keluarga": data_input[6],
            "Usia": data_input[7],
            "Prediksi": hasil,
            "Probabilitas": prob,
            "Faktor_Terkuat_1": top_features[0],
            "Faktor_Terkuat_2": top_features[1],
            "Faktor_Terkuat_3": top_features[2]
        }])

        os.makedirs("hasil", exist_ok=True)
        file_path = "hasil/hasil_prediksi.csv"
        if os.path.exists(file_path):
            hasil_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            hasil_df.to_csv(file_path, index=False)

        print("\n✅ Hasil prediksi disimpan di 'hasil/hasil_prediksi.csv'")
        return hasil_df

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
        return None