import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart City Accident ML Dashboard", layout="wide")

st.title("🚦 Smart City Accident Prediction & Insights")
st.write("An interactive Machine Learning dashboard for analyzing accident severity based on real-world smart city data.")

# =====================
# 📂 Upload CSV
# =====================
uploaded_file = st.file_uploader("📤 Upload your Accident Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # =====================
    # 🧹 Preprocessing
    # =====================
    st.markdown("### 🔧 Data Preprocessing")

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    st.success("✅ Categorical columns encoded successfully!")

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # =====================
    # 📊 EDA Visualizations
    # =====================
    st.markdown("### 📊 Exploratory Data Analysis")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # =====================
    # 🧠 Train-Test Split
    # =====================
    if 'Claim_Severity' not in df.columns:
        st.error("❌ 'Claim_Severity' column not found in the dataset!")
    else:
        X = df.drop('Claim_Severity', axis=1)
        y = df['Claim_Severity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # =====================
        # ⚙️ Model Training
        # =====================
        st.markdown("### 🤖 Model Training & Comparison")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
        }

        results = {}
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            trained_models[name] = model

        comparison_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
        st.subheader("🥇 Model Accuracy Comparison")
        st.bar_chart(comparison_df)

        best_model_name = max(results, key=results.get)
        best_model = trained_models[best_model_name]
        st.success(f"✅ Best Model: **{best_model_name}** with Accuracy: **{results[best_model_name]:.2f}**")

        # =====================
        # 🔢 User Input Prediction
        # =====================
        st.markdown("### 🧮 Predict Accident Severity Using Input Data")

        with st.form("prediction_form"):
            st.write("Enter input values for prediction:")

            input_data = {}
            for col in X.columns:
                value = st.number_input(f"{col}", value=float(df[col].mean()))
                input_data[col] = value

            submitted = st.form_submit_button("Predict Severity")

        if submitted:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = best_model.predict(input_scaled)
            st.success(f"🚨 Predicted Claim Severity: **{int(prediction[0])}**")

else:
    # =====================
    # 🚧 No CSV Uploaded → Manual Input Mode
    # =====================
    st.info("⬆️ No dataset uploaded. You can still try manual prediction below.")

    st.markdown("### ✍️ Enter Manual Input Data for Sample Prediction")

    with st.form("manual_input"):
        feature_1 = st.number_input("Speed (km/h)", min_value=0.0, value=60.0)
        feature_2 = st.number_input("Weather Condition (0=Clear, 1=Rain, 2=Fog)", min_value=0, max_value=2, value=0)
        feature_3 = st.number_input("Road Type (0=Urban, 1=Rural, 2=Highway)", min_value=0, max_value=2, value=1)
        feature_4 = st.number_input("Vehicle Age (years)", min_value=0.0, value=5.0)
        feature_5 = st.number_input("Driver Age (years)", min_value=16.0, value=30.0)

        submit_manual = st.form_submit_button("Predict Severity (Demo)")

    if submit_manual:
        # Simple demo prediction logic (no model yet)
        severity_score = (feature_1 * 0.03) + (feature_2 * 5) + (feature_3 * 2) + (feature_4 * 0.1)
        if severity_score > 50:
            result = "High Severity"
        elif severity_score > 25:
            result = "Medium Severity"
        else:
            result = "Low Severity"

        st.success(f"🚨 Predicted Accident Severity: **{result}**")

