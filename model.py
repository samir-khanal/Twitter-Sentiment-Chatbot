import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("cleaned_tweets.csv")
# Ensuring there is no NaN values in text column and replacing with empty string
df['cleaned_text'] = df['cleaned_text'].fillna("")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

from preprocessing import preprocess_tweet
# Function to predict sentiment
def predict_sentiment(user_input):
    cleaned_text = preprocess_tweet(user_input)
    # Converting to TF-IDF features
    transformed_text = vectorizer.transform([cleaned_text])
    # Predicting the sentiment
    prediction = model.predict(transformed_text)[0]

    sentiment_label = "Positive" if prediction == 1 else "Neutral" if prediction == 0 else "Negative"
    return sentiment_label

# Example chatbot interaction
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break
#     sentiment = predict_sentiment(user_input)
#     print(f"Chatbot: This tweet seems {sentiment}.")
