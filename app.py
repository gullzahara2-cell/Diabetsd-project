import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "Housing (2).csv"
MODEL_PATH = "house_price_model.pkl"
ENCODER_PATH = "lable_encoders.pkl"
FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]
CATEGORICAL_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def train_and_save_model():
    df = load_data()
    encoders = {}
    encoded_df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded_df[col] = encoder.fit_transform(encoded_df[col].astype(str))
        encoders[col] = encoder

    X = encoded_df[FEATURES]
    y = encoded_df["price"]

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return model, encoders


def load_model_and_encoders():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(ENCODER_PATH, "rb") as f:
            encoders = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, encoders

    return train_and_save_model()


model, encoders = load_model_and_encoders()

st.title("House Price Prediction")

area = st.number_input("Area", value=3000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=2)

mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.number_input("Parking", value=2)
prefarea = st.selectbox("Preferred Area", ["yes", "no"])

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"],
)

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }]
    )

    for col in CATEGORICAL_COLUMNS:
        input_df[col] = encoders[col].transform(input_df[col].astype(str))

    st.subheader("Input")
    st.dataframe(input_df)

    prediction = model.predict(input_df[FEATURES])[0]
    st.success(f"Predicted House Price: {prediction:,.2f}")