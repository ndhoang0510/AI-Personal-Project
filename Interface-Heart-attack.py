import streamlit as st
import numpy as np
import pandas as pd
import joblib

xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Dự đoán nguy cơ bị nhồi máu cơ tim")
st.write("Ứng dụng này sử dụng hai mô hình học máy: **XGBoost** và **Random Forest** để dự đoán.")

model_option = st.radio("Chọn mô hình dự đoán", ["XGBoost", "Random Forest", "So sánh cả hai"])

def get_user_input():
    data = {}

    data['age'] = st.slider("Tuổi", 18, 100, 45)
    data['waist_circumference'] = st.slider("Vòng eo (cm)", 50, 150, 85)
    data['sleep_hours'] = st.slider("Số giờ ngủ mỗi ngày", 0.0, 15.0, 7.0)
    data['blood_pressure_systolic'] = st.slider("Huyết áp tâm thu", 80, 200, 120)
    data['blood_pressure_diastolic'] = st.slider("Huyết áp tâm trương", 40, 130, 80)
    data['fasting_blood_sugar'] = st.slider("Đường huyết lúc đói", 60, 300, 100)
    data['cholesterol_level'] = st.slider("Tổng cholesterol", 100, 300, 200)
    data['cholesterol_hdl'] = st.slider("HDL", 20, 100, 50)
    data['cholesterol_ldl'] = st.slider("LDL", 50, 250, 130)
    data['triglycerides'] = st.slider("Triglycerides", 50, 400, 150)
    data['stress_level'] = st.selectbox("Mức độ căng thẳng", [0, 1, 2])

    data['hypertension'] = st.selectbox("Tăng huyết áp", [0, 1])
    data['diabetes'] = st.selectbox("Đái tháo đường", [0, 1])
    data['obesity'] = st.selectbox("Béo phì", [0, 1])
    data['family_history'] = st.selectbox("Tiền sử gia đình", [0, 1])
    data['previous_heart_disease'] = st.selectbox("Tiền sử bệnh tim", [0, 1])
    data['medication_usage'] = st.selectbox("Sử dụng thuốc", [0, 1])
    data['participated_in_free_screening'] = st.selectbox("Tham gia khám miễn phí", [0, 1])

    data['gender'] = st.selectbox("Giới tính", [0, 1])  # Male = 0, Female = 1
    data['region'] = st.selectbox("Khu vực sinh sống", [0, 1])  # Rural = 0, Urban = 1
    data['income_level'] = st.selectbox("Thu nhập", [0, 1, 2])  # Low, Middle, High
    data['smoking_status'] = st.selectbox("Hút thuốc", [0, 1, 2])  # Never, Past, Current
    data['alcohol_consumption'] = st.selectbox("Uống rượu", [0, 1, 2])  # None, Moderate, High
    data['physical_activity'] = st.selectbox("Vận động", [0, 1, 2])  # Low, Moderate, High
    data['dietary_habits'] = st.selectbox("Ăn uống lành mạnh", [0, 1])
    data['air_pollution_exposure'] = st.selectbox("Ô nhiễm không khí", [0, 1, 2])
    data['EKG_results'] = st.selectbox("Kết quả EKG", [0, 1])  # Normal = 0, Abnormal = 1

    return pd.DataFrame([data])

def compute_derived_features(df):
    df['bp_ratio'] = df['blood_pressure_systolic'] / df['blood_pressure_diastolic']
    df['sugar_triglyceride_ratio'] = df['fasting_blood_sugar'] / (df['triglycerides'] + 1)
    df['hdl_ldl_ratio'] = df['cholesterol_hdl'] / (df['cholesterol_ldl'] + 1)
    df['non_hdl_cholesterol'] = df['cholesterol_level'] - df['cholesterol_hdl']
    df['waist_age_ratio'] = df['waist_circumference'] / (df['age'] + 1)
    df['stress_sleep_ratio'] = df['stress_level'] / (df['sleep_hours'] + 0.1)
    df['metabolic_index'] = (df['fasting_blood_sugar'] + df['triglycerides'] + df['cholesterol_ldl']) / 3
    df['cholesterol_density'] = df['cholesterol_level'] / (df['waist_circumference'] + 1)
    return df

input_df = get_user_input()
input_df = compute_derived_features(input_df)
X_input = preprocessor.transform(input_df)

if st.button("Dự đoán"):
    xgb_prob = xgb_model.predict_proba(X_input)[0][1]
    rf_prob = rf_model.predict_proba(X_input)[0][1]

    if model_option == "XGBoost":
        st.subheader(f"Dự đoán (XGBoost): {xgb_prob * 100:.2f}% nguy cơ")
        st.warning("Cao" if xgb_prob >= 0.5 else "Thấp")

    elif model_option == "Random Forest":
        st.subheader(f"Dự đoán (Random Forest): {rf_prob * 100:.2f}% nguy cơ")
        st.warning("Cao" if rf_prob >= 0.5 else "Thấp")

    else:
        st.subheader("So sánh kết quả 2 mô hình:")
        st.write(f"XGBoost: {xgb_prob * 100:.2f}%")
        st.write(f"Random Forest: {rf_prob * 100:.2f}%")

        if xgb_prob >= 0.5 or rf_prob >= 0.5:
            st.error("Một trong hai mô hình đánh giá nguy cơ cao.")
        else:
            st.success("Cả hai mô hình đánh giá nguy cơ thấp.")
