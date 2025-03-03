import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
# Load dataset
df = pd.read_csv("Tweets.csv")
# checking dataset
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Droping unnecessary columns
df = df[['text', 'airline_sentiment']]
# Checking for missing values
print("Missing values:\n", df.isnull().sum())
# Mapping sentiment to numerical values
df['sentiment'] = df['airline_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})
df

# Function to clean tweets
def preprocess_tweet(text):
    # Convert text to lowercase
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    #Tokenize the text into words
    words = word_tokenize(text)
    #Remove stopwords; you can customize this list or use nltk's stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# applying preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_tweet)
df
# Saving cleaned dataset
df.to_csv("cleaned_tweets.csv", index=False)
print("Preprocessed data saved to cleaned_tweets.csv")