for lib in ["shap", "imblearn", "sklearn", "streamlit", "joblib", "pandas", "numpy", "matplotlib", "seaborn"]:
    try:
        __import__(lib)
        print(f"{lib}: available")
    except ImportError:
        print(f"{lib}: NOT available")
