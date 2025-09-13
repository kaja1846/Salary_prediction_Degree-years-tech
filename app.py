import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# ----------------------------
# Build Pipeline
# ----------------------------
def build_pipeline(random_state=42):
    numeric_features = ["years_experience"]
    numeric_transformer = Pipeline([("scaler", StandardScaler())])

    categorical_features = ["degree"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    text_transformer = TfidfVectorizer(
        max_features=50, token_pattern=r"(?u)\b[^, ]+\b", lowercase=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("deg", categorical_transformer, categorical_features),
            ("tech", text_transformer, "technologies"),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=150, random_state=random_state, n_jobs=-1
            )),
        ]
    )
    return model

# ----------------------------
# Train on data
# ----------------------------
def train_model(df):
    X = df[["years_experience", "degree", "technologies"]]
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, score

# ----------------------------
# Sample Dataset
# ----------------------------
def load_sample_data():
    data = {
        "years_experience": [2, 5, 8, 12, 3, 7, 15, 4, 10, 1],
        "degree": [
            "Bachelors", "Masters", "Bachelors", "PhD", "Diploma",
            "Masters", "Bachelors", "Bachelors", "PhD", "Diploma"
        ],
        "technologies": [
            "python,sql", "python,aws,azure", "java,spark,sql", "python,sql,tableau,aws", "excel,powerbi",
            "python,tableau,sql", "java,spark,aws", "python,sql", "python,sql,aws,docker", "excel"
        ],
        "salary": [35000, 75000, 95000, 135000, 28000, 90000, 140000, 45000, 125000, 25000],
    }
    return pd.DataFrame(data)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("💰 Salary Prediction App")
st.write("Predict salaries based on years of experience, degree, and technologies.")

uploaded_file = st.file_uploader("📂 Upload your CSV (years_experience, degree, technologies, salary)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset.")
    df = load_sample_data()

st.write("### Training Data Preview")
st.dataframe(df.head())

# Train Model
with st.spinner("Training model..."):
    model, score = train_model(df)
    joblib.dump(model, "salary_model.pkl")

st.success(f"✅ Model trained! R² Score on validation: {score:.2f}")

# ----------------------------
# Prediction Form
# ----------------------------
st.write("### 🔮 Predict Salary")

with st.form("prediction_form"):
    years_exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=3)
    degree = st.selectbox("Degree", options=["Diploma", "Bachelors", "Masters", "PhD"])
    technologies = st.text_input("Technologies (comma-separated)", value="python,sql")

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    model = joblib.load("salary_model.pkl")
    input_df = pd.DataFrame(
        [[years_exp, degree, technologies]],
        columns=["years_experience", "degree", "technologies"],
    )

    prediction = model.predict(input_df)[0]
    st.success(f"💵 Predicted Salary: **${prediction:,.2f}**")