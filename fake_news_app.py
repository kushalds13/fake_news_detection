import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# App title
st.title("📰 Fake News Detector")

# App description
st.write("🔍 Enter a news article below to check if it's Fake or Real!")

# Text input
news_text = st.text_area("Enter News Text Here:")

# Predict button
if st.button("Predict"):
    if news_text:
        # Transform the input
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)

        # Display result
        if prediction[0] == 0:
            st.error("🔴 This news is likely FAKE!")
        else:
            st.success("🟢 This news is likely REAL!")
    else:
        st.warning("⚠️ Please enter some text.")