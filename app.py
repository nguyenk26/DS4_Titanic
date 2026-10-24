import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    with st.spinner("Đang tải mô hình Titanic, vui lòng chờ..."):
        return joblib.load(CURRENT_DIR / "titanic_model.pkl")


def build_input_data(pclass, sex, age, fare, sibsp, parch, embarked, title):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    has_cabin = 0

    return pd.DataFrame([
        {
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
            "HasCabin": has_cabin,
        }
    ])


if "model" not in st.session_state:
    st.session_state.model = load_model()

model = st.session_state.model

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Dự đoán Sống sót Tàu Titanic")
st.write("Nhập thông tin hành khách dưới đây để dự đoán khả năng sống sót.")

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

if st.button("Dự đoán kết quả"):
    input_data = build_input_data(pclass, sex, age, fare, sibsp, parch, embarked, title)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Khả năng SỐNG SÓT cao! (Xác suất: {probability*100:.2f}%)")
    else:
        st.error(f"Khả năng CHẾT cao! (Xác suất sống sót: {probability*100:.2f}%)")
