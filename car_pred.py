import pandas as pd
import streamlit as st
import datetime
import pickle
import sklearn

st.header('Cars 24 Price Prediction App')

df = pd.read_csv("/Users/suraaj/Desktop/StremlitTutorial/cars24-car-price.csv")

# st.dataframe(df)

fuel_type = st.selectbox(
    "Select Fuel Type: ",
    ("Petrol", "Diesel", "CNG", "Electric",  "LPG")
)

engine = st.slider("Set the Engine Power", 500, 5000, step=100)

col1, col2= st.columns(2)

with col1:
    transmission_type = st.selectbox(
        "Select Transmission Type",
        ("Manual", "Automatic")

    )

with col2:
    seats = st.selectbox(
        "Select no. of seats ", [4, 5, 6, 7, 8])


encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
    "transmission_type": {"Manual": 1, "Automatic": 2}
}

def model_pred (fuel_encoded,transmission_encoded, seats, engine):
    with open("car_pred (1)", "rb") as file:
        reg_model = pickle.load(file)

        input_features = [[2018.0, 1, 120000, fuel_encoded, transmission_encoded, 19.7, engine, 46.3, seats]]
        return reg_model.predict(input_features)



if st.button("Predict"):
    fuel_encoded= encode_dict['fuel_type'][fuel_type]
    transmission_encoded= encode_dict['transmission_type'][transmission_type]

    price = model_pred (fuel_encoded,transmission_encoded, seats, engine)
    st.text("Predicted Price is" + str(price))

