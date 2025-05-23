# src/eksplorasi_data.py

import pandas as pd

def tampilkan_informasi_dataset(df):
    print("\n--- INFORMASI DATASET ---")
    print(f"Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print("\n5 data pertama:")
    print(df.head())
    
    print("\nStatistik deskriptif:")
    print(df.describe().round(2))
    
    print("\nJumlah nilai kosong per kolom:")
    print(df.isnull().sum())
    
    print("\nDistribusi kelas target:")
    print(df["Outcome"].value_counts())
    print(f"Persentase Diabetes: {df['Outcome'].mean()*100:.2f}%")


def cek_nilai_nol_tidak_valid(df):
    """
    Mengecek kolom yang tidak seharusnya memiliki nilai nol.
    Contohnya: Glukosa, Tekanan Darah, dan BMI.
    """
    print("\nJumlah nilai nol per kolom (yang mungkin tidak valid):")
    kolom_tidak_valid = ['Glucose', 'BloodPressure', 'BMI']
    for kolom in kolom_tidak_valid:
        jumlah_nol = (df[kolom] == 0).sum()
        if jumlah_nol > 0:
            print(f"- {kolom}: {jumlah_nol} nilai nol")
