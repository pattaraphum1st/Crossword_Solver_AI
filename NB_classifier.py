import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

# File paths
MODEL_FILE = "crossword_naive_bayes.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
DATA_FILE = "C:/Users/USER/Downloads/cleaned_optimized_times.csv"

# Function to preprocess text
def clean_text(text):
    """Lowercase, remove special characters, stopwords, and lemmatize."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Function to train and save the model
def train_model():
    print("Training model...")

    # Load dataset
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=['answer', 'optimized_clue'])

    # Keep only frequent words
    word_counts = df['answer'].value_counts()
    common_words = word_counts[word_counts >= 10].index
    df = df[df['answer'].isin(common_words)]  

    # Clean clues
    df['cleaned_clue'] = df['optimized_clue'].apply(clean_text)

    # Convert clues into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000, max_df=0.75, min_df=15)
    X = vectorizer.fit_transform(df['cleaned_clue'])

    # Encode answers into integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['answer'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = ComplementNB()
    clf.fit(X_train, y_train)

    # Save trained model
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)

    # Evaluate accuracy
    y_pred = clf.predict(csr_matrix(X_test))  # Convert to sparse matrix
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained and saved! Accuracy: {accuracy:.2f}")

# Function to load the model
def load_model():
    global clf, vectorizer, label_encoder
    clf = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    print("Model loaded successfully!")

# Function to classify a word from a clue
def classify_word(clue, pattern):
    clue = clean_text(clue)
    clue_vector = vectorizer.transform([clue])
    predicted_index = clf.predict(csr_matrix(clue_vector))[0]  # Use sparse matrix
    predicted_word = label_encoder.inverse_transform([predicted_index])[0]

    possible_words = find_matching_words(pattern)

    if possible_words:
        if predicted_word in possible_words:
            return predicted_word
        else:
            return "Not found"

    return predicted_word  

# Function to find words matching a pattern
def find_matching_words(pattern):
    regex_pattern = pattern.replace('_', '.')  
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=['answer'])

    matches = [word for word in df['answer'] if re.fullmatch(regex_pattern, word, re.IGNORECASE)]
    matches = list(set(matches))
    return matches

# Check if model exists, otherwise train
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
    train_model()
else:
    load_model()

# Example usage
print(classify_word("Stay above water", "__E"))
