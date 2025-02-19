import streamlit as st
from model import predict_sentiment

# Streamlit UI
st.title("Twitter Sentiment Chatbot 🤖")
st.write("Type a tweet and get its sentiment!")

user_input = st.text_input("Enter your tweet:")
if st.button("Analyze"):
    sentiment = predict_sentiment(user_input)
    st.write(f"**Sentiment:** {sentiment}")