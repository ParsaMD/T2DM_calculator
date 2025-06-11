import pandas as pd
import joblib
import streamlit as st

# ----------------- LOAD MODELS -----------------
scaler = joblib.load("std_scaler_7param.pkl")
cal_svm = joblib.load("calibrated_svm_7param.pkl")
cal_stack = joblib.load("calibrated_stacking_7param.pkl")

# Input variable order
input_order = ['FBS', 'BMI', 'Age', 'Sex', 'FH1_Diabetes', 'SESq', 'CVD']
# Output variables
output_order = ['z_FBS', 'z_BMI', 'z_Age', 'Sex', 'FH1_Diabetes', 'SESq', 'CVD']

# Define risk categories 
def categorize(prob):
    if prob < 0.10:
        return 'Low'
    elif prob < 0.20:
        return 'Medium'
    else:
        return 'High'

st.title("T2DM Risk Calculator (7-Parameter ML)")

st.write("Input your values below to estimate risk of incident type 2 diabetes using machine learning (SVM and Stacking Ensemble). All fields are required.")

with st.form("t2dm_form"):
    fbs = st.number_input("Baseline Fasting blood glucose (mg/dL)", min_value=60, max_value=126)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=13.0, max_value=60.0)
    age = st.number_input("Age (years)", min_value=15, max_value=100)
    sex = st.selectbox("Sex", options=[("Female",0),("Male",1)], format_func=lambda x: x[0])
    fh1 = st.selectbox("First-degree family history of diabetes",options=[("No",0),("Yes",1)], format_func=lambda x: x[0])
    ses = st.selectbox("Socioeconomic Status quartile (e.g. 1=Lowest, 5=Highest)", options=[1,2,3,4,5])
    cvd = st.selectbox("History of cardiovascular disease",options=[("No",0),("Yes",1)], format_func=lambda x: x[0])
    submit = st.form_submit_button("Predict Risk")

if submit:
    # 1. Collect into DataFrame
    inp = pd.DataFrame([{
        "FBS":fbs, "BMI":bmi, "Age":age,
        "Sex":sex[1], "FH1_Diabetes":fh1[1],
        "SESq":ses, "CVD":cvd[1]
    }])

    # 2. Standardize as TRAIN scaler
    inp[["z_FBS", "z_BMI", "z_Age"]] = scaler.transform(inp[["FBS", "BMI", "Age"]])
    X_final = inp[output_order]   # ensure column order

    # 3. Predict
    prob_svm = float(cal_svm.predict_proba(X_final)[0][1])
    prob_stack = float(cal_stack.predict_proba(X_final)[0][1])

    st.subheader("Prediction Results")
    st.write(f"**SVM Calibrated Probability:** {prob_svm*100:.1f}% ({categorize(prob_svm)} risk)")
    st.write(f"**Stacking Ensemble Calibrated Probability:** {prob_stack*100:.1f}% ({categorize(prob_stack)} risk)")
    st.caption("Risk categories are illustrative. Interpretation and use should be done in consultation with your clinician.")
    st.caption("Developed by Parsa Amirian M.D.")
