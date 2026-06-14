import json

cells = [
    # Cell 0: Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Heart Disease Risk Prediction using UCI Cleveland Dataset\n",
            "\n",
            "This notebook implements an end-to-end Machine Learning project to predict the risk of heart disease based on patient clinical data from the UCI Cleveland dataset. \n",
            "\n",
            "### Project Workflow:\n",
            "1. **Data Loading & Cleaning**: Download and load the processed Cleveland dataset; handle missing values (`?`).\n",
            "2. **Exploratory Data Analysis (EDA)**: Understand distribution of clinical features and their relationship to heart disease risk.\n",
            "3. **Preprocessing & Encoding**: Apply dummy variable encoding to nominal categoricals and map the target variable to binary.\n",
            "4. **Train-Test Split & Standardization**: Scale all numerical and dummy features.\n",
            "5. **SMOTE**: Balance class representations in the training partition to avoid bias.\n",
            "6. **Model Training & Hyperparameter Tuning**: Use `GridSearchCV` on a `LogisticRegression` model.\n",
            "7. **Evaluation**: Compute Accuracy, ROC-AUC, classification reports, and plot the ROC Curve.\n",
            "8. **Explainability Analysis**: Employ **SHAP** (SHapley Additive exPlanations) to identify and interpret clinical risk factors."
        ]
    },
    # Cell 1: Imports
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.model_selection import train_test_split, GridSearchCV\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, plot_roc_curve\n",
            "from imblearn.over_sampling import SMOTE\n",
            "import joblib\n",
            "import shap\n",
            "import urllib.request\n",
            "import os\n",
            "\n",
            "# Set visualization styles\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
            "print(\"All libraries imported successfully!\")"
        ]
    },
    # Cell 2: Section 1 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Dataset\n",
            "We download the classic `processed.cleveland.data` directly from the UCI Machine Learning Repository."
        ]
    },
    # Cell 3: Download and load data
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data\"\n",
            "filename = \"processed.cleveland.data\"\n",
            "\n",
            "if not os.path.exists(filename):\n",
            "    print(f\"Downloading dataset from {url}...\")\n",
            "    urllib.request.urlretrieve(url, filename)\n",
            "    print(\"Download complete.\")\n",
            "else:\n",
            "    print(\"Dataset already exists locally.\")\n",
            "\n",
            "column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', \n",
            "                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']\n",
            "\n",
            "# Load with na_values='?' because missing values in this file are marked as '?'\n",
            "df = pd.read_csv(filename, header=None, names=column_names, na_values='?')\n",
            "print(f\"Dataset Loaded successfully. Shape: {df.shape}\")"
        ]
    },
    # Cell 4: View head
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.head()"
        ]
    },
    # Cell 5: Section 2 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Exploratory Data Analysis (EDA) & Data Cleaning"
        ]
    },
    # Cell 6: Data Info
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df.info()"
        ]
    },
    # Cell 7: Check Nulls
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Missing Values:\")\n",
            "print(df.isnull().sum())"
        ]
    },
    # Cell 8: Handle Missing Values
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Since the missing entries are very minimal (4 in ca, 2 in thal), we impute them using their modes\n",
            "df['ca'].fillna(df['ca'].mode()[0], inplace=True)\n",
            "df['thal'].fillna(df['thal'].mode()[0], inplace=True)\n",
            "\n",
            "print(\"Missing Values after Imputation:\")\n",
            "print(df.isnull().sum())"
        ]
    },
    # Cell 9: Target distribution plot
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Map num to a binary classification: 0 (No Heart Disease) vs 1 (Presence of Heart Disease)\n",
            "df['target'] = (df['num'] > 0).astype(int)\n",
            "print(\"Target Variable Distribution:\")\n",
            "print(df['target'].value_counts())\n",
            "\n",
            "sns.countplot(x='target', data=df, palette='Set2')\n",
            "plt.title('Distribution of Target Variable')\n",
            "plt.xticks([0, 1], ['No Disease (0)', 'Disease (1)'])\n",
            "plt.show()"
        ]
    },
    # Cell 10: Numerical distributions
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
            "sns.histplot(ax=axes[0, 0], data=df, x='age', hue='target', kde=True, multiple='stack', palette='muted')\n",
            "axes[0, 0].set_title('Age Distribution by Target')\n",
            "\n",
            "sns.histplot(ax=axes[0, 1], data=df, x='trestbps', hue='target', kde=True, multiple='stack', palette='muted')\n",
            "axes[0, 1].set_title('Resting Blood Pressure Distribution by Target')\n",
            "\n",
            "sns.histplot(ax=axes[1, 0], data=df, x='chol', hue='target', kde=True, multiple='stack', palette='muted')\n",
            "axes[1, 0].set_title('Serum Cholesterol Distribution by Target')\n",
            "\n",
            "sns.histplot(ax=axes[1, 1], data=df, x='thalach', hue='target', kde=True, multiple='stack', palette='muted')\n",
            "axes[1, 1].set_title('Max Heart Rate Distribution by Target')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 11: Categorical distributions
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
            "sns.countplot(ax=axes[0, 0], data=df, x='sex', hue='target', palette='Set2')\n",
            "axes[0, 0].set_title('Heart Disease by Sex (0 = Female, 1 = Male)')\n",
            "\n",
            "sns.countplot(ax=axes[0, 1], data=df, x='cp', hue='target', palette='Set2')\n",
            "axes[0, 1].set_title('Heart Disease by Chest Pain Type (cp)')\n",
            "\n",
            "sns.countplot(ax=axes[1, 0], data=df, x='thal', hue='target', palette='Set2')\n",
            "axes[1, 0].set_title('Heart Disease by Thalassemia Type (thal)')\n",
            "\n",
            "sns.countplot(ax=axes[1, 1], data=df, x='ca', hue='target', palette='Set2')\n",
            "axes[1, 1].set_title('Heart Disease by Number of Major Vessels (ca)')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    # Cell 12: Correlation Heatmap
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
            "plt.title('Correlation Matrix of Cleveland Features')\n",
            "plt.show()"
        ]
    },
    # Cell 13: Section 3 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Preprocessing and Feature Encoding\n",
            "We drop the original `num` column, convert categorical fields to integers, and one-hot encode nominal columns (`cp`, `restecg`, `slope`, `thal`)."
        ]
    },
    # Cell 14: Dummy variables
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Drop original num column\n",
            "df.drop('num', axis=1, inplace=True)\n",
            "\n",
            "# Cast categories to integers\n",
            "for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:\n",
            "    df[col] = df[col].astype(int)\n",
            "\n",
            "# Apply one-hot encoding on nominal features\n",
            "categorical_cols = ['cp', 'restecg', 'slope', 'thal']\n",
            "df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n",
            "\n",
            "print(\"Encoded Feature Columns:\")\n",
            "print(df_encoded.columns.tolist())\n",
            "print(f\"Encoded dataset shape: {df_encoded.shape}\")"
        ]
    },
    # Cell 15: Section 4 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Train-Test Split and Standardization"
        ]
    },
    # Cell 16: Train Test Split and Scale
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X = df_encoded.drop('target', axis=1)\n",
            "y = df_encoded['target']\n",
            "\n",
            "# Stratified train-test split (80% train, 20% test)\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "# Fit StandardScaler on training features and scale\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "print(f\"Training size: {X_train_scaled.shape[0]} samples\")\n",
            "print(f\"Test size: {X_test_scaled.shape[0]} samples\")"
        ]
    },
    # Cell 17: Section 5 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Address Class Imbalance using SMOTE\n",
            "We employ SMOTE (Synthetic Minority Over-sampling Technique) to ensure training classes are fully balanced."
        ]
    },
    # Cell 18: Apply SMOTE
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Before SMOTE in training:\")\n",
            "print(y_train.value_counts())\n",
            "\n",
            "smote = SMOTE(random_state=42)\n",
            "X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)\n",
            "\n",
            "print(\"\\nAfter SMOTE in training:\")\n",
            "print(pd.Series(y_train_res).value_counts())"
        ]
    },
    # Cell 19: Section 6 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Train and Fine-Tune Logistic Regression using GridSearchCV"
        ]
    },
    # Cell 20: GridSearchCV
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "param_grid = {\n",
            "    'C': [0.001, 0.01, 0.1, 1, 10, 100],\n",
            "    'penalty': ['l1', 'l2'],\n",
            "    'solver': ['liblinear', 'saga']\n",
            "}\n",
            "\n",
            "lr = LogisticRegression(max_iter=10000, random_state=42)\n",
            "\n",
            "# Run grid search sequentially to prevent Loky/Winapi multiprocessing bugs in Python 3.7.0\n",
            "grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc')\n",
            "grid_search.fit(X_train_res, y_train_res)\n",
            "\n",
            "best_model = grid_search.best_estimator_\n",
            "print(\"Optimal Hyperparameters:\", grid_search.best_params_)"
        ]
    },
    # Cell 21: Section 7 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Model Evaluation"
        ]
    },
    # Cell 22: Score and report
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "y_pred = best_model.predict(X_test_scaled)\n",
            "y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "accuracy = accuracy_score(y_test, y_pred)\n",
            "roc_auc = roc_auc_score(y_test, y_pred_proba)\n",
            "\n",
            "print(f\"Accuracy: {accuracy * 100:.2f}%\")\n",
            "print(f\"ROC-AUC: {roc_auc:.4f}\")\n",
            "\n",
            "print(\"\\nClassification Report:\")\n",
            "print(classification_report(y_test, y_pred))\n",
            "\n",
            "print(\"\\nConfusion Matrix:\")\n",
            "cm = confusion_matrix(y_test, y_pred)\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Disease'], yticklabels=['Normal', 'Disease'])\n",
            "plt.ylabel('True label')\n",
            "plt.xlabel('Predicted label')\n",
            "plt.title('Confusion Matrix')\n",
            "plt.show()"
        ]
    },
    # Cell 23: Plot ROC
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot ROC Curve using the scikit-learn helper\n",
            "plot_roc_curve(best_model, X_test_scaled, y_test)\n",
            "plt.title('Receiver Operating Characteristic (ROC) Curve')\n",
            "plt.plot([0, 1], [0, 1], 'r--')\n",
            "plt.show()"
        ]
    },
    # Cell 24: Section 8 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Explainability Analysis using SHAP\n",
            "We employ SHAP to explain how the trained Logistic Regression model makes its predictions, and identify key clinical risk factors."
        ]
    },
    # Cell 25: Initialize SHAP
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# We initialize a SHAP LinearExplainer\n",
            "explainer = shap.LinearExplainer(best_model, X_train_res, feature_perturbation=\"interventional\")\n",
            "shap_values = explainer(X_test_scaled)\n",
            "\n",
            "print(\"SHAP value calculations complete.\")"
        ]
    },
    # Cell 26: Summary plot
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Global interpretation: Summary Plot\n",
            "# This plot shows the feature importance and impact direction. \n",
            "# Notice how chest pain type (cp_4) and thalassemia indicators (thal_7) play critical roles.\n",
            "shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)"
        ]
    },
    # Cell 27: Bar plot
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Global interpretation: Mean SHAP values (feature importance bar plot)\n",
            "shap.plots.bar(shap_values)"
        ]
    },
    # Cell 28: Section 9 Header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Save Preprocessing and Model Artifacts"
        ]
    },
    # Cell 29: Save joblib files
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save the trained model, scaler, and columns list using Joblib\n",
            "joblib.dump(best_model, 'logistic_regression_heart.pkl')\n",
            "joblib.dump(scaler, 'scaler.pkl')\n",
            "joblib.dump(X.columns.tolist(), 'columns.pkl')\n",
            "print(\"Model, scaler, and column names successfully saved for web application deployment!\")"
        ]
    }
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Heart_disease_risk.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generation complete!")
