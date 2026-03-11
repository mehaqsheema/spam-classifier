import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Spam Message Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):

    transformed_sms = vectorizer.transform([input_sms])

    result = model.predict(transformed_sms)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")