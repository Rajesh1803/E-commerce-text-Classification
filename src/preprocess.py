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

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def word_cloud(df):
    """Generates word clouds for each category."""
    categories = df["category"].unique()
    for category in categories:
        text_data = " ".join(df[df["category"] == category]["text"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {category}")
        plt.show()

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

def preprocess_data(file_path, max_features=5000, test_size=0.2, n_components=300):
    """
    Loads, cleans, vectorizes, applies LSA for decorrelation, and splits data.

    Args:
        file_path (str): Path to dataset CSV file.
        max_features (int): Max words for TF-IDF vectorization.
        test_size (float): Proportion for test set.
        n_components (int): Number of LSA components.

    Returns:
        tuple: Processed training and test data.
    """
    logging.info("Loading dataset...")
    df = pd.read_csv(file_path, names=["category", "text"])
    df.dropna(subset=["category", "text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)

    logging.info(f"Dataset size after cleaning: {df.shape[0]} rows")

    # Clean text
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Generate word clouds (Optional)
    # word_cloud(df)

    # TF-IDF Vectorization
    logging.info("Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df["cleaned_text"])

    # Apply LSA to reduce direct correlation with target
    logging.info("Applying Latent Semantic Analysis (LSA)...")
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsa = lsa.fit_transform(X_tfidf)

    # Target labels
    y = df["category"]

    # Splitting dataset
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_lsa, y, test_size=test_size, random_state=42)

    logging.info("Preprocessing complete.")

    return X_train, X_test, y_train, y_test, vectorizer, lsa

if __name__ == "__main__":
    file_path = "../Data/ecommerceDataset.csv"
    X_train, X_test, y_train, y_test, vectorizer, lsa = preprocess_data(file_path)

    logging.info("Data preprocessing is complete and ready for model training.")
