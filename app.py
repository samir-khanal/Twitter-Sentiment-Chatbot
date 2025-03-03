import streamlit as st
from model import predict_sentiment
import nltk
import os

# ✅ Ensure NLTK uses the correct directory for downloads
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# ✅ Force download the correct resource
nltk.download('punkt', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('stopwords', download_dir=os.path.join(os.getcwd(), "nltk_data"))

# Streamlit UI
st.title("Twitter Sentiment Chatbot 🤖")
st.write("Type a tweet and get its sentiment!")

user_input = st.text_input("Enter your tweet:")
if st.button("Analyze"):
    sentiment = predict_sentiment(user_input)
    st.write(f"**Sentiment:** {sentiment}")