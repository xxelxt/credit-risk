import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

@st.cache_data
def load_resources():
    model = joblib.load("joblib/XGBoost_0.3_GridSearch.joblib")
    scaler = joblib.load("joblib/scaler.pkl")
    label_mapping = joblib.load("joblib/label_encoder.pkl")
    return model, scaler, label_mapping

model, scaler, label_mapping = load_resources()

def transform(input_data, scaler, label_mapping):
    bins_dict = {
        "age": [20, 30, 40, 50, 60],
        "no_of_children": [-1, 0, 1, 2, 10],
        "net_yearly_income": [0, 100000, 200000, 500000, float('inf')],
        "no_of_days_employed": [-1, 1000, 3000, 6000, float('inf')],
        "total_family_members": [0, 1, 3, float('inf')],
        "yearly_debt_payments": [0, 10000, 25000, 50000, float('inf')],
        "credit_limit": [0, 50000, 100000, 200000, float('inf')],
        "credit_limit_used(%)": [-1, 25, 50, 75, 100],
        "credit_score": [0, 600, 700, 800, 900, float('inf')],
        "prev_defaults": [-1, 0, 1, float('inf')]
    }

    labels_dict = {
        "age": ['20-29', '30-39', '40-49', '50-59'],
        "no_of_children": ['0', '1', '2', '3+'],
        "net_yearly_income": ['<100,000', '100,000-200,000', '200,000-500,000', '>500,000'],
        "no_of_days_employed": ['<1,000', '1,000-3,000', '3,000-6,000', '>6,000'],
        "total_family_members": ['1', '2-3', '4+'],
        "yearly_debt_payments": ['<10,000', '10,000-25,000', '25,000-50,000', '>50,000'],
        "credit_limit": ['<50,000', '50,000-100,000', '100,000-200,000', '>200,000'],
        "credit_limit_used(%)": ['0-25%', '25-50%', '50-75%', '75-100%'],
        "credit_score": ['<600', '600-700', '700-800', '800-900', '>900'],
        "prev_defaults": ['0', '1', '2+']
    }

    for col in bins_dict.keys():
        input_data[col] = pd.cut(input_data[col], bins=bins_dict[col], labels=labels_dict[col])

    for col in ['occupation_type'] + list(bins_dict.keys()):
        input_data[col] = input_data[col].map(lambda x: {v: k for k, v in label_mapping[col].items()}.get(x))

    input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == "Nam" else 0)
    
    binary_features = ['owns_car', 'owns_house', 'migrant_worker', 'default_in_last_6months']
    for feature in binary_features:
        input_data[feature] = input_data[feature].apply(lambda x: 1 if x == "Có" else 0)

    input_data = input_data.astype({col: 'int64' for col in input_data.select_dtypes('object').columns})

    continuous_columns = [
        'age', 'net_yearly_income', 'no_of_days_employed',
        'yearly_debt_payments', 'credit_limit', 
        'credit_limit_used(%)', 'credit_score'
    ]
    
    input_data[continuous_columns] = scaler.transform(input_data[continuous_columns])

    return input_data

st.title("Dự đoán rủi ro tín dụng")

with st.sidebar:
    st.title("Thông tin khách hàng")
    age = st.slider("Tuổi", min_value=18, max_value=65, value=30)
    col1, col2 = st.sidebar.columns(2)

    with col1:
        gender = st.radio("Giới tính", ["Nam", "Nữ"])
        migrant_worker = st.radio("Lao động di cư?", ["Có", "Không"])
    
    with col2:
        owns_car = st.radio("Sở hữu xe?", ["Có", "Không"])
        owns_house = st.radio("Sở hữu nhà?", ["Có", "Không"])
        
    no_of_children = st.slider("Số con", min_value=0, max_value=10, value=0)
    total_family_members = st.slider("Số thành viên trong gia đình", min_value=1, max_value=10, value=3)
    st.markdown("---")

    occupation_type = st.selectbox("Loại công việc", [
        "Unknown", "Laborers", "Sales staff", "Core staff", "Managers", 
        "Drivers", "High skill tech staff", "Accountants", "Medicine staff", 
        "Security staff", "Cooking staff", "Cleaning staff", "Private service staff", 
        "Low-skill Laborers", "Secretaries", "Waiters/barmen staff", 
        "Realty agents", "HR staff", "IT staff"
    ])
    
    no_of_days_employed = st.number_input("Số ngày làm việc", min_value=2, max_value=365252, value=200, format="%d")
    net_yearly_income = st.number_input("Thu nhập hàng năm", min_value=0, max_value=2217660, value=500000, format="%d")
    yearly_debt_payments = st.number_input("Số tiền thanh toán nợ hàng năm", min_value=2000, max_value=280000, value=5000, format="%d")
    st.markdown("---")
    
    credit_limit = st.number_input("Hạn mức tín dụng", min_value=0, max_value=648100, value=150000, format="%d")
    credit_limit_used = st.slider("Phần trăm sử dụng hạn mức", min_value=0, max_value=100, value=20)
    credit_score = st.slider("Điểm tín dụng", min_value=400, max_value=950, value=600)
    
    prev_defaults = st.slider("Số lần vỡ nợ trước", min_value=0, max_value=10, value=0)
    default_in_last_6months = st.radio("Có vỡ nợ trong 6 tháng gần nhất?", ["Có", "Không"])

input_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "owns_car": [owns_car],
    "owns_house": [owns_house],
    "no_of_children": [no_of_children],
    "net_yearly_income": [net_yearly_income],
    "no_of_days_employed": [no_of_days_employed],
    "occupation_type": [occupation_type],
    "total_family_members": [total_family_members],
    "migrant_worker": [migrant_worker],
    "yearly_debt_payments": [yearly_debt_payments],
    "credit_limit": [credit_limit],
    "credit_limit_used(%)": [credit_limit_used],
    "credit_score": [credit_score],
    "prev_defaults": [prev_defaults],
    "default_in_last_6months": [default_in_last_6months]
})

input_data = transform(input_data, scaler, label_mapping)

predicted_probabilities = model.predict_proba(input_data)[:, 1]
prediction = model.predict(input_data)[0]
risk = "Rủi ro cao" if prediction >= 0.5 else "Rủi ro thấp"

st.write(f"<p style='font-size: 20px;'><b>Xác suất rủi ro tín dụng:</b> {predicted_probabilities[0]:.5f} <br><b>Đánh giá rủi ro:</b> {risk}</p>", unsafe_allow_html=True)

st.subheader("Thông tin đã nhập:")
st.json(input_data.to_dict(orient='records'))