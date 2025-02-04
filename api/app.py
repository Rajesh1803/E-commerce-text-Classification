from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  # LSA for decorrelation
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = FastAPI()

# Load the trained model and vectorizer
model = joblib.load('src/models/model.pkl')
vectorizer = joblib.load('src/models/vectorizer.pkl')
lsa=joblib.load('src/models/lsa.pkl')
def clean_text(text):
    """
    Cleans text by removing special characters, punctuation, stopwords, and applying lemmatization.
    """
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase and remove punctuation
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization and lemmatization
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)


class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    """
    Endpoint to predict the category of a given text.
    :param request: The request containing the text.
    :return: Predicted category
    """
    text = request.text
    text=clean_text(text)
    # print(1)
    # Transform the text using the vectorizer
    text_vectorized = vectorizer.transform([text])
    text_vectorized=lsa.transform(text_vectorized)
    # Make prediction
    prediction = model.predict(text_vectorized)
    categories = ['Electronics', 'Household', 'Books', 'Clothing & Accessories']
    if isinstance(prediction[0], int):  # If it's an index
        return categories[prediction[0]]
    else:  # If it's a label
        return prediction[0]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
