# src/menu.py
from src.evaluasi_model import simpan_evaluasi
from src.muat_data import muat_dataset
from src.eksplorasi_data import tampilkan_informasi_dataset, cek_nilai_nol_tidak_valid
from src.preprocessing import tangani_nilai_nol, normalisasi_fitur
from src.pelatihan_model import bagi_data, latih_model
from src.evaluasi_model import evaluasi_model
from src.visualisasi import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_distribusi_kelas,
    plot_korelasi
)
from src.prediksi import prediksi_data_baru
import os
import pandas as pd

def menu_utama():
    print("="*50)
    print("PREDIKSI DIABETES MENGGUNAKAN REGRESI LOGISTIK")
    print("="*50)

    # Muat dan proses data
    df = muat_dataset()
    tampilkan_informasi_dataset(df)
    cek_nilai_nol_tidak_valid(df)

    df_processed = tangani_nilai_nol(df)
    X = df_processed.drop("Outcome", axis=1)
    y = df_processed["Outcome"]
    feature_names = X.columns
    X_scaled, scaler = normalisasi_fitur(X)
    X_train, X_test, y_train, y_test = bagi_data(X_scaled, y)
    model = latih_model(X_train, y_train, scaler)

    hasil_evaluasi = evaluasi_model(model, X_test, y_test)
    simpan_evaluasi(hasil_evaluasi)

    while True:
        print("\n" + "="*50)
        print("MENU UTAMA")
        print("="*50)
        print("1. Prediksi Diabetes untuk Data Baru")
        print("2. Lihat Evaluasi Model")
        print("3. Lihat Visualisasi")
        print("4. Informasi Dataset")
        print("5. Lihat Hasil Prediksi Sebelumnya")
        print("6. Keluar")

        pilihan = input("\nPilih menu (1-6): ")

        if pilihan == '1':
            prediksi_data_baru(model, scaler, feature_names)

        elif pilihan == '2':
            print("\n--- EVALUASI MODEL ---")
            print(f"Akurasi: {hasil_evaluasi['akurasi']:.4f}")
            print(f"ROC AUC Score: {hasil_evaluasi['roc_auc']:.4f}")
            print(f"\nMatriks Konfusi:\n{hasil_evaluasi['confusion_matrix']}")
            print(f"\nLaporan Klasifikasi:\n{hasil_evaluasi['classification_report']}")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == '3':
            print("\n--- MENAMPILKAN VISUALISASI ---")
            plot_confusion_matrix(hasil_evaluasi["confusion_matrix"])
            plot_roc_curve(y_test, hasil_evaluasi["y_pred_proba"])
            plot_feature_importance(model, feature_names)
            plot_distribusi_kelas(df)
            plot_korelasi(df)
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == '4':
            print("\n--- INFORMASI DATASET ---")
            print(f"Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")
            print(f"Fitur: {', '.join(feature_names)}")
            print(f"Distribusi kelas: {df['Outcome'].value_counts().to_dict()}")
            print(f"Persentase Diabetes: {df['Outcome'].mean()*100:.2f}%")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == '5':
            print("\n--- HASIL PREDIKSI SEBELUMNYA ---")
            file_path = "hasil/hasil_prediksi.csv"
            if os.path.exists(file_path):
                df_prediksi = pd.read_csv(file_path)
                print(df_prediksi.tail(5).to_string(index=False))
            else:
                print("Belum ada hasil prediksi yang tersimpan.")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == '6':
            print("\nTerima kasih telah menggunakan aplikasi prediksi diabetes!")
            break
        else:
            print("❌ Pilihan tidak valid. Silakan masukkan angka 1-6.")
