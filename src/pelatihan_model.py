# src/pelatihan_model.py

import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

def bagi_data(X, y, test_size=0.2, random_state=42):
    """
    Membagi data menjadi data latih dan uji.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def latih_model(X_train, y_train, scaler, simpan_model=True,
                model_path='model/model_diabetes.pkl',
                scaler_path='model/scaler_diabetes.pkl'):
    """
    Melatih model dengan hyperparameter tuning menggunakan GridSearchCV.
    Jika model sudah ada, akan dimuat ulang.
    """
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        print("✅ Model dimuat dari file yang sudah ada.")
    else:
        print("Melatih model baru dengan GridSearchCV...")
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        grid = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy'
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"✅ Model selesai dilatih. Parameter terbaik: {grid.best_params_}")

        if simpan_model:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print("✅ Model dan scaler disimpan ke file.")
    
    return model
