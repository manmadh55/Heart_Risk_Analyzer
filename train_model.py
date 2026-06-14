import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import shap
import urllib.request
import os

def main():
    # 1. Download/Load UCI Cleveland Dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    filename = "processed.cleveland.data"
    
    if not os.path.exists(filename):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv(filename, header=None, names=column_names, na_values='?')
    
    # 2. Preprocess & Clean Missing Values
    # Impute missing values with mode (since missing rows are minimal: ca has 4, thal has 2)
    df['ca'].fillna(df['ca'].mode()[0], inplace=True)
    df['thal'].fillna(df['thal'].mode()[0], inplace=True)
    
    # Convert nominal/categorical variables to integers for dummy generation
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        df[col] = df[col].astype(int)
        
    # 3. Binary classification mapping (target: 0 = normal, 1 = heart disease)
    df['target'] = (df['num'] > 0).astype(int)
    df.drop('num', axis=1, inplace=True)
    
    # One-hot encode categorical features (cp, restecg, slope, thal)
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    # 4. Train-Test Split (80/20) with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. SMOTE for Class Imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    # 7. GridSearchCV for Hyperparameter Tuning of Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(max_iter=10000, random_state=42)
    # n_jobs=1 to avoid python 3.7.0 loky win32 _winapi bug on Windows
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train_res, y_train_res)
    
    best_model = grid_search.best_estimator_
    
    # 8. Evaluation
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n=== Model Training Summary ===")
    print("Best Parameters found:", grid_search.best_params_)
    print(f"Accuracy on Test Set: {acc * 100:.2f}%")
    print(f"ROC-AUC on Test Set: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # 9. SHAP Explainability verification
    explainer = shap.LinearExplainer(best_model, X_train_res, feature_dependence="independent")
    shap_values = explainer(X_test_scaled)
    print("SHAP explainer successfully initialized. SHAP values shape:", shap_values.shape)
    
    # 10. Save Artifacts for Streamlit App
    # We save the model, the scaler, and the column structure
    joblib.dump(best_model, 'logistic_regression_heart.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'columns.pkl')
    print("\nArtifacts saved successfully:")
    print(" - logistic_regression_heart.pkl")
    print(" - scaler.pkl")
    print(" - columns.pkl")

if __name__ == '__main__':
    main()
