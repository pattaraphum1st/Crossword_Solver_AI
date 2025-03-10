import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
df = pd.read_csv("C:/Users/USER/Downloads/cleaned_optimized_times.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Preprocess text: lowercase, remove special characters, stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Function to classify a word from a given clue
def classify_word(clue, pattern):
    """Predicts a word based on the given clue and matches it with the given pattern."""
    clue = clean_text(clue)
    clue_vector = vectorizer.transform([clue])
    predicted_word = clf.predict(clue_vector)[0]
    
    # Find words that match the pattern
    possible_words = find_matching_words(pattern)
    if possible_words:
        if predicted_word in possible_words:
            return print(predicted_word)
        else:
            return possible_words[0]  # Return first matching word
    
    return print(predicted_word)  # Return prediction if no matches found

# Function to match words based on a missing letter pattern
def find_matching_words(pattern):
    """Find words that match a given pattern (e.g., M_R_ _T -> matches MERCAT)."""
    regex_pattern = pattern.replace('_', '.')  # Convert '_' to regex wildcard '.'
    matches = [word for word in df['answer'].dropna() if re.fullmatch(regex_pattern, word, re.IGNORECASE)]
    return matches


def load_model():
    """Load the trained classifier and vectorizer."""
    global clf, vectorizer
    clf = joblib.load('crossword_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

# Example usage
load_model()
print(classify_word("Animal doctor tense after sheep rejected", "M_R_ _T"))
