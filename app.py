


import streamlit as st
import pandas as pd
import pickle
import joblib
import time
import os

# ------------------------------------------------------------
# 🎯 Load Trained XGBoost Model (with Safe Handling)
# ------------------------------------------------------------
MODEL_PATH = "xgboost_model.pkl"

def load_model(model_path):
    """Safely load an XGBoost model using pickle or joblib."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: `{model_path}`")
        st.stop()

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
            st.success("✅ Model loaded successfully using Pickle.")
            return model
    except Exception as e1:
        st.warning(f"⚠️ Pickle load failed: {e1}")
        try:
            model = joblib.load(model_path)
            st.success("✅ Model loaded successfully using Joblib.")
            return model
        except Exception as e2:
            st.error(f"❌ Model loading failed with both Pickle and Joblib.\n\nError: {e2}")
            st.stop()

model = load_model(MODEL_PATH)

# ------------------------------------------------------------
# 🧭 Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="🏦 Smart Loan Approval Predictor",
    layout="wide",
    page_icon="💰"
)

# ------------------------------------------------------------
# 🎨 Custom Styling
# ------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #F9FAFB;
        color: #111827;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🏠 Header Section
# ------------------------------------------------------------
st.title("💰 Loan Approval Prediction System")
st.markdown("""
Welcome to the **Smart Loan Approval Predictor** — powered by **XGBoost**.  
Enter applicant details to see if their loan is likely to be **approved or rejected**.
""")
st.divider()

# ------------------------------------------------------------
# 🎛️ Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("📘 About the App")
    st.write("""
    This app uses a **Machine Learning (XGBoost)** model trained on real loan applicant data.
    It predicts whether a loan will be approved based on key financial and demographic features.
    """)
    st.info("💡 Tip: Good credit history, high income, and semiurban properties improve approval odds!")

# ------------------------------------------------------------
# 🧍 Input Section
# ------------------------------------------------------------
st.subheader("📝 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    Education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Property_Area = st.selectbox("Property Area", ("Rural", "Semiurban", "Urban"))

with col2:
    ApplicantIncome = st.number_input("Applicant Income (₹)", min_value=0, step=500)
    CoapplicantIncome = st.number_input("Coapplicant Income (₹)", min_value=0, step=500)
    LoanAmount = st.number_input("Loan Amount (in thousands ₹)", min_value=0, step=10)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0, step=12)
    Credit_History = st.selectbox("Credit History (1=Good, 0=Bad)", (1.0, 0.0))

# ------------------------------------------------------------
# 🧮 Data Encoding
# ------------------------------------------------------------
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}

property_encoded = {
    'Property_Area_Rural': 1 if Property_Area == 'Rural' else 0,
    'Property_Area_Semiurban': 1 if Property_Area == 'Semiurban' else 0,
    'Property_Area_Urban': 1 if Property_Area == 'Urban' else 0
}

input_data = {
    'Gender': gender_map[Gender],
    'Married': married_map[Married],
    'Dependents': dependents_map[Dependents],
    'Education': education_map[Education],
    'Self_Employed': self_employed_map[Self_Employed],
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    **property_encoded
}

input_df = pd.DataFrame([input_data])

# ------------------------------------------------------------
# 🔮 Prediction Section
# ------------------------------------------------------------
st.divider()
st.subheader("🔍 Prediction Result")

if st.button("🚀 Predict Loan Status"):
    with st.spinner("Analyzing your application..."):
        time.sleep(1.2)
        try:
            prediction = int(model.predict(input_df)[0])
            confidence = model.predict_proba(input_df)[0][prediction] * 100

            if prediction == 1:
                st.success(f"🎉 Loan Approved! ✅ (Confidence: {confidence:.2f}%)")
                st.markdown("""
                💡 **Insight:**  
                Approval likely due to a strong credit history or solid repayment ability.
                """)
            else:
                st.error(f"🚫 Loan Rejected (Confidence: {confidence:.2f}%)")
                st.markdown("""
                ⚠️ **Possible Factors:**  
                - Low or missing credit history  
                - High loan-to-income ratio  
                - Property area or dependent count affecting eligibility
                """)

            with st.expander("📋 View Encoded Input Data"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")
            st.info("Ensure input features match the model's training data structure.")

# ------------------------------------------------------------
# 📊 Insights
# ------------------------------------------------------------
st.divider()
st.subheader("📈 Loan Approval Tips")
st.markdown("""
✅ **Approval Factors:**  
- Good **credit history (1.0)** significantly increases chances.  
- **Semiurban properties** show the highest approval rates.  
- Balanced **income-to-loan ratio** improves eligibility.  
- **Fewer dependents** may indicate better repayment potential.  

💡 **Try This:**  
Adjust the loan amount or term and observe how it changes the approval outcome.
""")
