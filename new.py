import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background("Heart.png")


disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.fillna(disease_df.mean(), inplace=True)

selected_features = ['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']
X = disease_df[selected_features]
y = disease_df['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)


scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


logreg = LogisticRegression(class_weight="balanced", random_state=4)
logreg.fit(X_train_scaled, y_train)


st.markdown("""
     <style>
           
    /* Neon background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 80px; /* Change size as needed */
        font-weight: bold;
        text-align: center;
        color: #fff;
        text-shadow: 0 0 10px #0ff, 0 0 20px #0ff, 0 0 40px #00f, 0 0 80px #00f;
        margin-bottom: 20px;
    }
     /* Neon text for headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00ffcc;
        text-shadow: 0 0 5px #00ffcc, 0 0 10px #00ffcc, 0 0 20px #00ffcc;
    }
    
    .stButton>button {
        background-color: #111;
        border-radius: 8px;
        color: white;
        border: 2px solid transparent;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.4s ease;
    }
    .stButton>button:hover {
        background-color: #111;
        border: 2px solid #ff00ff;
        box-shadow: 0px 0px 20px 5px rgba(255, 0, 255, 0.7);
        color: #ff00ff;
        transform: scale(1.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("   ❤️ Heart Disease Prediction ❤️")

st.write("This app predicts the risk of heart disease (CHD) based on the given features.")


age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex (Male = 1, Female = 0)", [1, 0])
cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=220)
sysBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=130)
glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=85)


if st.button('Predict'):
    patient_data = pd.DataFrame([[age, sex, cigsPerDay, totChol, sysBP, glucose]], columns=selected_features)
    patient_scaled = scaler.transform(patient_data)

    prediction = logreg.predict(patient_scaled)[0]
    prediction_prob = logreg.predict_proba(patient_scaled)[0][1]

    if prediction == 1:
        result = f"👉 **Risk of CHD: Yes** with a probability of **{prediction_prob:.2%}**"
    else:
        result = f"👉 **Risk of CHD: No** with a probability of **{prediction_prob:.2%}**"
    
    st.write(result)
