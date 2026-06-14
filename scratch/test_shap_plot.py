import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load artifacts
model = joblib.load('logistic_regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Load background data
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv('processed.cleveland.data', header=None, names=column_names, na_values='?')
df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)
for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
    df[col] = df[col].astype(int)
df['target'] = (df['num'] > 0).astype(int)
df.drop('num', axis=1, inplace=True)
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df_encoded.drop('target', axis=1)
X_scaled_bg = scaler.transform(X)

# Create a sample input
raw_input = {
    'age': 55,
    'sex': 1,
    'trestbps': 130,
    'chol': 250,
    'fbs': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'ca': 0,
    'cp_2': 0,
    'cp_3': 0,
    'cp_4': 1,
    'restecg_1': 0,
    'restecg_2': 1,
    'slope_2': 1,
    'slope_3': 0,
    'thal_6': 0,
    'thal_7': 1
}
input_df = pd.DataFrame([raw_input])
input_df = input_df[columns]
scaled_input = scaler.transform(input_df)

# Run explainer
explainer = shap.LinearExplainer(model, X_scaled_bg, feature_perturbation="interventional")
shap_values = explainer(scaled_input)

# Plot waterfall
fig = plt.figure(figsize=(10, 5))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.savefig('scratch/shap_test_waterfall.png')
print("SHAP waterfall plot successfully generated and saved!")
