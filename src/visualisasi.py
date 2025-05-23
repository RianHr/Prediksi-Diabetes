# src/visualisasi.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import os

# Set gaya visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(style="darkgrid")
plt.rcParams.update({'font.size': 12})


def simpan_gambar(path):
    """
    Simpan plot ke file, pastikan folder tujuan ada.
    """
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(path)


def plot_confusion_matrix(cm, save_path="visualisasi/confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks([0.5, 1.5], ['Tidak Diabetes', 'Diabetes'])
    plt.yticks([0.5, 1.5], ['Tidak Diabetes', 'Diabetes'])
    plt.tight_layout()
    simpan_gambar(save_path)
    plt.show()


def plot_roc_curve(y_test, y_score, save_path="visualisasi/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    simpan_gambar(save_path)
    plt.show()


def plot_feature_importance(model, feature_names, save_path="visualisasi/feature_importance.png"):
    coef = model.coef_[0]
    importance = pd.DataFrame({
        'Fitur': feature_names,
        'Koefisien': coef,
        'Penting': np.abs(coef)
    }).sort_values('Penting', ascending=False)
    
    plt.figure(figsize=(10, 6))
    warna = ['red' if c < 0 else 'green' for c in importance['Koefisien']]
    plt.bar(importance['Fitur'], importance['Koefisien'], color=warna)
    plt.xticks(rotation=45, ha='right')
    plt.title("Koefisien Fitur dalam Model Regresi Logistik")
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    simpan_gambar(save_path)
    plt.show()
    
    return importance


def plot_distribusi_kelas(df, save_path="visualisasi/distribusi_kelas.png"):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Outcome', palette='Set2')
    plt.title('Distribusi Pasien Diabetes vs Tidak Diabetes')
    plt.xticks([0, 1], ['Tidak Diabetes', 'Diabetes'])
    plt.xlabel('Status Diabetes')
    plt.ylabel('Jumlah Pasien')
    plt.tight_layout()
    simpan_gambar(save_path)
    plt.show()


def plot_korelasi(df, save_path="visualisasi/korelasi.png"):
    plt.figure(figsize=(12, 10))
    korelasi = df.corr()
    mask = np.triu(np.ones_like(korelasi, dtype=bool))
    sns.heatmap(korelasi, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
    plt.title('Matriks Korelasi antar Fitur')
    plt.tight_layout()
    simpan_gambar(save_path)
    plt.show()
