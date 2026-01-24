import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details and get survival probability.")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", ["male", "female"])
    pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=28.0, step=1.0)
    fare = st.number_input("Fare", min_value=0.0, value=30.0, step=1.0)

with col2:
    sibsp = st.number_input("SibSp (siblings/spouses)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parch (parents/children)", min_value=0, max_value=10, value=0, step=1)
    has_cabin = st.checkbox("Has cabin?", value=False)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], index=0)

cabin = "X" if has_cabin else ""  # pipeline сам превращает в Has_Cabin_Number

if st.button("Predict"):
    raw_input = {
        "Sex": sex,
        "Age": float(age),
        "Fare": float(fare),
        "Pclass": int(pclass),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Cabin": cabin,
        "Title": title,
    }

    out = predict(raw_input)

    st.subheader("Result")
    st.write(f"**Survival probability:** {out['percent']}")
    st.write(f"**Threshold (F1-optimized):** {out['threshold_f1']:.4f}")

    label = "✅ Survived" if out["pred"] == 1 else "❌ Did not survive"
    st.write(f"**Prediction:** {label}")
