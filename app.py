
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# 1. Load model đã đóng gói
CURRENT_DIR = Path(__file__).resolve().parent
model = joblib.load(CURRENT_DIR / 'titanic_model.pkl')

st.title("🚢 Dự đoán Sống sót Tàu Titanic")
st.write("Nhập thông tin hành khách dưới đây để dự đoán khả năng sống sót.")

# 2. Giao diện nhập liệu
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Giới tính", ["male", "female"])
    pclass = st.selectbox("Hạng vé (Pclass)", [1, 2, 3])
    age = st.slider("Tuổi", 1, 80, 28)
    fare = st.number_input("Giá vé (Fare)", min_value=0.0, value=32.0)

with col2:
    sibsp = st.number_input("Số anh chị em/vợ chồng đi cùng (SibSp)", min_value=0, value=0)
    parch = st.number_input("Số bố mẹ/con cái đi cùng (Parch)", min_value=0, value=0)
    embarked = st.selectbox("Cảng lên tàu (Embarked)", ["S", "C", "Q"])
    title = st.selectbox("Danh xưng (Title)", ["Mr", "Mrs", "Miss", "Master", "Rare"])

# 3. Tính toán các feature phụ giống như bước Feature Engineering
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
has_cabin = 0 # Giả định mặc định

# Tạo dataframe đầu vào
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Title": title,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "HasCabin": has_cabin
}])

# 4. Dự đoán khi bấm nút
if st.button("Dự đoán kết quả"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"Khả năng SỐNG SÓT cao! (Xác suất: {probability*100:.2f}%)")
    else:
        st.error(f"Khả năng CHẾT cao! (Xác suất sống sót: {probability*100:.2f}%)")
