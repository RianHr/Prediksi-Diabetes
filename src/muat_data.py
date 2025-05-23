# src/muat_data.py

import pandas as pd
import os
import sys

def muat_dataset(path="data/diabetes.csv"):
    """
    Fungsi untuk memuat dataset diabetes dari path yang ditentukan.

    Parameter:
        path (str): Lokasi file CSV.

    Return:
        df (DataFrame): Data yang dimuat.
    """
    if not os.path.exists(path):
        print("❌ File dataset tidak ditemukan.")
        print(f"   Pastikan file '{path}' ada di folder 'data/'")
        print("   Download dataset di: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        sys.exit(1)
    else:
        print("✅ Dataset berhasil dimuat")
        df = pd.read_csv(path)
        return df
