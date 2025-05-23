# src/evaluasi_model.py

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def evaluasi_model(model, X_test, y_test):
    """
    Melakukan evaluasi terhadap model menggunakan data uji.

    Return:
        Dictionary berisi metrik evaluasi dan prediksi.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    akurasi = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- EVALUASI MODEL ---")
    print(f"Akurasi: {akurasi:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nMatriks Konfusi:\n", cm)
    print("\nLaporan Klasifikasi:\n", report)

    return {
        "akurasi": akurasi,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }

def simpan_evaluasi(hasil, filepath='hasil/laporan_evaluasi.txt'):
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("=== EVALUASI MODEL ===\n")
        f.write(f"Akurasi: {hasil['akurasi']:.4f}\n")
        f.write(f"ROC AUC Score: {hasil['roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(hasil['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(hasil['classification_report'])
    print(f"\n✅ Hasil evaluasi disimpan di: {filepath}")
