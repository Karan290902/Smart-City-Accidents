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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
    st.caption("#Insight: Correlation heatmap shows how strongly each feature affects accident severity.")

    col1, col2 = st.columns(2)
    with col1:
        if 'Weather' in df.columns and 'Claim_Severity' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='Weather', hue='Claim_Severity', data=df, ax=ax)
            plt.xticks(rotation=30)
            st.pyplot(fig)
            st.caption("#Insight: Rainy or foggy conditions show higher accident severity levels.")

    with col2:
        if 'Road_Condition' in df.columns and 'Claim_Amount' in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='Road_Condition', y='Claim_Amount', data=df, ax=ax)
            plt.xticks(rotation=30)
            st.pyplot(fig)
            st.caption("#Insight: Damaged roads correlate with higher claim amounts.")

    # =====================
    # 🧠 Train-Test Split
    # =====================
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

        st.subheader(f"📈 {name}")
        st.write("Accuracy:", round(acc, 3))
        st.text(classification_report(y_test, y_pred))
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)
        st.caption(f"#Insight: {name} model confusion matrix shows its prediction power and misclassification rate.")

        if name != "Logistic Regression":
            feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            feat_imp.head(10).plot(kind='bar', ax=ax)
            plt.title(f"Top 10 Important Features - {name}")
            st.pyplot(fig)
            st.caption("#Insight: Displays which features most influence accident claim severity.")

    # =====================
    # 🧩 Model Comparison
    # =====================
    st.markdown("### 🥇 Model Accuracy Comparison")
    comparison_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
    st.bar_chart(comparison_df)
    best_model_name = comparison_df['Accuracy'].idxmax()
    best_model = trained_models[best_model_name]
    st.success(f"🏆 Best Model: {best_model_name} with Accuracy = {results[best_model_name]:.3f}")

    # =====================
    # ✋ User Input for Prediction
    # =====================
    st.markdown("### ✋ Predict Accident Severity Using Custom Input")

    user_input = {}
    st.info("Enter values for each feature below:")
    for col in X.columns:
        user_input[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()) if np.issubdtype(df[col].dtype, np.number) else 0.0)

    if st.button("🔍 Predict Severity"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = best_model.predict(input_scaled)
        st.success(f"🎯 Predicted Accident Severity: **{prediction[0]}**")

else:
    st.info("⬆️ Please upload your dataset to start the analysis.")
