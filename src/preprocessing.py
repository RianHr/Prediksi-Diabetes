# src/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def tangani_nilai_nol(df):
    """
    Mengganti nilai nol yang tidak valid dengan NaN,
    lalu mengisi NaN menggunakan median berdasarkan kelas Outcome.
    """
    df_processed = df.copy()
    kolom_nol = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for kolom in kolom_nol:
        df_processed[kolom] = df_processed[kolom].replace(0, np.nan)
        df_processed[kolom] = df_processed.groupby('Outcome')[kolom].transform(
            lambda x: x.fillna(x.median())
        )

    # Jika masih ada NaN (fallback)
    if df_processed.isnull().sum().sum() > 0:
        df_processed = df_processed.fillna(df_processed.median())

    return df_processed


def normalisasi_fitur(X):
    """
    Melakukan standarisasi terhadap fitur.
    Mengembalikan fitur yang telah distandarisasi dan objek scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
