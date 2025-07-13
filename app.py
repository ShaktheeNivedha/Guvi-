import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🩺 Breast Cancer Prediction App")

# Load model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# List of all 30 features used during training
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select input method", ["Manual Entry", "Upload CSV File"])

# ========== Manual Input ==========
if input_method == "Manual Entry":
    st.subheader("🔘 Manual Input (30 features)")

    input_data = []
    for feature in features:
        value = st.number_input(f"{feature}", min_value=0.0, step=0.01, format="%.4f")
        input_data.append(value)

    input_array = np.array(input_data).reshape(1, -1)

    if st.button("Predict"):
        try:
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            result = "Malignant" if prediction[0] == 1 else "Benign"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error: {e}")

# ========== CSV File Upload ==========
elif input_method == "Upload CSV File":
    st.subheader("📁 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with 30 features", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                st.write("✅ Uploaded Data Preview:")
                st.dataframe(df.head())

                input_scaled = scaler.transform(df[features])
                predictions = model.predict(input_scaled)
                df['Prediction'] = ["Malignant" if p == 1 else "Benign" for p in predictions]

                st.success("Batch Prediction Completed ✅")
                st.dataframe(df[['Prediction']])

                # Download results
                result_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", result_csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
