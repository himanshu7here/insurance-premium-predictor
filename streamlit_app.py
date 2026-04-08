import streamlit as st
import pickle
import numpy as np

# 🔹 Page configuration
st.set_page_config(page_title="Insurance Predictor", layout="centered")

# 🔹 Load model
model = pickle.load(open('model.pkl', 'rb'))

# 🔹 Title & Description
st.markdown("<h1 style='text-align: center;'>💡 Insurance Premium Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")
st.info("This app predicts insurance premium based on health and lifestyle factors.")

# 🔹 User Inputs
st.subheader("📋 Enter Your Details")

age = st.slider("Age", 18, 65)

diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
bp = st.selectbox("Blood Pressure Problems (0 = No, 1 = Yes)", [0, 1])
transplant = st.selectbox("Any Transplants (0 = No, 1 = Yes)", [0, 1])
chronic = st.selectbox("Chronic Disease (0 = No, 1 = Yes)", [0, 1])

height = st.number_input("Height (cm)", min_value=100.0)
weight = st.number_input("Weight (kg)", min_value=30.0)

allergy = st.selectbox("Known Allergies (0 = No, 1 = Yes)", [0, 1])
cancer = st.selectbox("Family Cancer History (0 = No, 1 = Yes)", [0, 1])

surgeries = st.slider("Number of Major Surgeries", 0, 3)

# 🔹 Feature Engineering
bmi = weight / ((height / 100) ** 2)
st.write(f"📊 Calculated BMI: {bmi:.2f}")

risk_score = diabetes + bp + chronic + transplant + surgeries

# 🔹 Prediction
if st.button("🚀 Predict Premium"):

    # Feature order MUST match training
    features = np.array([[age, diabetes, bp, transplant, chronic,
                          height, weight, allergy, cancer, surgeries,
                          bmi, risk_score]])

    prediction = model.predict(features)

    st.success(f"💰 Estimated Premium: ₹{int(prediction[0])}")

    # 🔹 Interpretation
    if prediction[0] > 30000:
        st.warning("⚠️ High premium detected. This indicates higher health risk.")
    elif prediction[0] > 20000:
        st.info("ℹ️ Moderate premium. Maintain a healthy lifestyle.")
    else:
        st.success("✅ Low premium. Good health profile!")