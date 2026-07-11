import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

st.title("Diabetes Prediction Explorer")
st.write("Use this app to explore the diabetes dataset and train a simple classifier.")

data = load_data()

st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show raw dataset", value=True)
show_summary = st.sidebar.checkbox("Show summary statistics", value=True)
show_chart = st.sidebar.checkbox("Show feature distributions", value=False)

if show_data:
    st.subheader("Raw data")
    st.dataframe(data)

if show_summary:
    st.subheader("Dataset summary")
    st.write(data.describe())
    st.write("Outcome distribution:")
    st.bar_chart(data["Outcome"].value_counts())

if show_chart:
    st.subheader("Feature distributions")
    selected_feature = st.selectbox(
        "Select feature", [col for col in data.columns if col != "Outcome"]
    )
    st.write(data[selected_feature].describe())
    st.bar_chart(data[selected_feature].value_counts())

st.sidebar.header("Model training")
with st.sidebar.form(key="model_form"):
    test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    n_estimators = st.slider("Random Forest trees", min_value=10, max_value=200, value=100, step=10)
    submit = st.form_submit_button("Train model")

if submit:
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model results")
    st.write(f"Accuracy: {accuracy:.3f}")
    st.text("Classification report:")
    st.text(classification_report(y_test, y_pred, digits=3))

    st.subheader("Feature importances")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importance)
