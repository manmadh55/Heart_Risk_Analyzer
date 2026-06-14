import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Risk Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient */
    .header-container {
        background: linear-gradient(135deg, #FF4B4B 0%, #850000 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 75, 75, 0.2);
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Premium Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #FF4B4B;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid rgba(255, 75, 75, 0.1);
        padding-bottom: 0.5rem;
    }
    
    /* Styled Results */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        margin-top: 1rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .result-high {
        background: linear-gradient(135deg, #ffe3e3 0%, #ffc5c5 100%);
        border: 2px solid #FF4B4B;
        color: #850000;
    }
    
    .result-low {
        background: linear-gradient(135deg, #e3ffe5 0%, #c5ffc9 100%);
        border: 2px solid #2ECC71;
        color: #0c5912;
    }
    
    /* Custom Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF4B4B 0%, #D32F2F 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
        background: linear-gradient(135deg, #FF6B6B 0%, #E53935 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load machine learning artifacts
@st.cache_resource
def load_ml_assets():
    model = joblib.load("logistic_regression_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, columns = load_ml_assets()
except Exception as e:
    st.error("⚠️ Error loading model artifacts. Make sure to run `train_model.py` first.")
    st.stop()

# Load reference background dataset for SHAP
@st.cache_data
def load_shap_bg():
    filename = "processed.cleveland.data"
    if not os.path.exists(filename):
        return None
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = pd.read_csv(filename, header=None, names=column_names, na_values='?')
    df['ca'].fillna(df['ca'].mode()[0], inplace=True)
    df['thal'].fillna(df['thal'].mode()[0], inplace=True)
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        df[col] = df[col].astype(int)
    df['target'] = (df['num'] > 0).astype(int)
    df.drop('num', axis=1, inplace=True)
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop('target', axis=1)
    # Scale background data
    X_scaled = scaler.transform(X)
    return X_scaled, X.columns.tolist()

X_scaled_bg, encoded_cols = load_shap_bg()

# Header Layout
st.markdown("""
    <div class="header-container">
        <div class="header-title">❤️ Heart Disease Risk Predictor</div>
        <div class="header-subtitle">Advanced machine learning diagnostic tool utilizing the UCI Cleveland database and SHAP explainability.</div>
    </div>
""", unsafe_allow_html=True)

st.write("Please input the patient's clinical measurements below to evaluate heart disease risk and view diagnostic feature explanations.")

# Main Form Split Layout
with st.form("patient_data_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-title">👤 Patient Profile & Vitals</div>', unsafe_allow_html=True)
        age = st.slider("Patient Age", 1, 120, 50, help="Age in years")
        sex = st.selectbox("Biological Sex", ["Female", "Male"], index=1, help="Patient biological sex")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120, step=1, help="Resting blood pressure on admission")
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200, step=1, help="Serum cholesterol level")
        
    with col2:
        st.markdown('<div class="section-title">⚡ Stress Test & ECG</div>', unsafe_allow_html=True)
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150, step=1, help="Max heart rate recorded during cardiac stress test")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest")
        slope = st.selectbox(
            "Slope of ST Segment",
            ["Upsloping", "Flat", "Downsloping"],
            index=1,
            help="Slope of the peak exercise ST segment"
        )
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"], index=0, help="Chest pain triggered by exercise stress test")
        
    with col3:
        st.markdown('<div class="section-title">🩺 Lab Results & Cardiac Features</div>', unsafe_allow_html=True)
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            index=3,
            help="Classification of chest pain symptoms"
        )
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["False", "True"], index=0, help="Blood sugar level before eating")
        restecg = st.selectbox(
            "Resting ECG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            index=0,
            help="Electrocardiographic assessment at rest"
        )
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3], index=0, help="Fluoroscopy count of major blood vessels (0-3)")
        thal = st.selectbox(
            "Thalassemia Type (thal)",
            ["Normal", "Fixed Defect", "Reversible Defect"],
            index=0,
            help="Inherited blood disorder screening result"
        )
        
    submit_button = st.form_submit_button("Perform Diagnostics")

# Execute prediction and interpretability when form is submitted
if submit_button:
    # 1. Map Streamlit inputs to numerical feature mappings
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "True" else 0
    exang_val = 1 if exang == "Yes" else 0
    
    # One-hot encoding manual maps
    cp_map = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}
    cp_val = cp_map[cp]
    
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg_val = restecg_map[restecg]
    
    slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
    slope_val = slope_map[slope]
    
    thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
    thal_val = thal_map[thal]
    
    # 2. Build feature representation matching training
    raw_input = {
        'age': age,
        'sex': sex_val,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_val,
        'thalach': thalach,
        'exang': exang_val,
        'oldpeak': oldpeak,
        'ca': ca,
        'cp_2': 1 if cp_val == 2 else 0,
        'cp_3': 1 if cp_val == 3 else 0,
        'cp_4': 1 if cp_val == 4 else 0,
        'restecg_1': 1 if restecg_val == 1 else 0,
        'restecg_2': 1 if restecg_val == 2 else 0,
        'slope_2': 1 if slope_val == 2 else 0,
        'slope_3': 1 if slope_val == 3 else 0,
        'thal_6': 1 if thal_val == 6 else 0,
        'thal_7': 1 if thal_val == 7 else 0
    }
    
    input_df = pd.DataFrame([raw_input])
    # Align columns to exactly match training order
    input_df = input_df[columns]
    
    # 3. Standardize and Scale
    scaled_input = scaler.transform(input_df)
    
    # 4. Predict Risk Probability
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    
    # Show Diagnostic Result Card
    st.markdown('<div class="section-title">📊 Diagnostic Results</div>', unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.markdown(f"""
                <div class="result-card result-high">
                    <h2 style='margin:0;'>⚠️ High Risk</h2>
                    <p style='font-size: 1.1rem; margin: 0.5rem 0;'>The model indicates presence of Heart Disease.</p>
                    <h1 style='font-size: 3.5rem; margin: 0;'>{prob*100:.1f}%</h1>
                    <p style='font-size: 0.9rem; opacity: 0.8; margin:0;'>Risk Probability Score</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card result-low">
                    <h2 style='margin:0;'>✅ Low Risk</h2>
                    <p style='font-size: 1.1rem; margin: 0.5rem 0;'>The model indicates absence of Heart Disease.</p>
                    <h1 style='font-size: 3.5rem; margin: 0;'>{prob*100:.1f}%</h1>
                    <p style='font-size: 0.9rem; opacity: 0.8; margin:0;'>Risk Probability Score</p>
                </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        # Progress Bar visualizer
        st.write("### Risk Score Indicator")
        st.progress(prob)
            
        st.write(
            "This assessment is generated by a Logistic Regression model fine-tuned using GridSearchCV "
            "and trained on a SMOTE-balanced representation of the UCI Cleveland dataset. "
            "This model achieved **~85.3% Accuracy** and **~0.95 ROC-AUC** in test evaluations."
        )
        
    st.markdown("---")
    
    # 5. SHAP Explainability Plot
    st.markdown('<div class="section-title">🔍 SHAP Explainability Analysis</div>', unsafe_allow_html=True)
    st.write(
        "The plot below shows how the patient's individual clinical factors contributed to the prediction. "
        "Factors in **red (positive SHAP)** push the prediction risk higher, while factors in **blue (negative SHAP)** "
        "lower the risk score. The base value ($f(x)$ offset) represents the model's average prediction across the database."
    )
    
    if X_scaled_bg is not None:
        with st.spinner("Generating SHAP waterfall plot..."):
            try:
                explainer = shap.LinearExplainer(model, X_scaled_bg, feature_perturbation="interventional")
                shap_values = explainer(scaled_input)
                
                # Plot SHAP waterfall
                fig, ax = plt.subplots(figsize=(10, 5.5))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                plt.title("SHAP Explanation for Current Patient Diagnostic", fontsize=12, fontweight='bold', pad=15)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not render SHAP explanation: {e}")
    else:
        st.info("ℹ️ SHAP background reference data not found. Please verify `processed.cleveland.data` is in the workspace.")