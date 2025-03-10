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

# Drop rows where 'answer' or 'optimized_clue' is missing
df = df.dropna(subset=['answer', 'optimized_clue'])

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Preprocess text: lowercase, remove special characters, stopwords, and lemmatize."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply text cleaning
df['cleaned_clue'] = df['optimized_clue'].apply(clean_text)

# Convert clues to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=50000, max_df=0.95, min_df=2)  # Ignore very common/rare words
X = vectorizer.fit_transform(df['cleaned_clue'])

# Set target variable (correcting from optimized_clue to answer)
y = df['answer']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = SGDClassifier(loss='log_loss', random_state=42)
clf.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(clf, 'crossword_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Predict on test data and evaluate performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

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
            return predicted_word
        else:
            return possible_words[0]  # Return first matching word
    
    return predicted_word  # Return prediction if no matches found

# Function to match words based on a missing letter pattern
def find_matching_words(pattern):
    """Find words that match a given pattern (e.g., M_R_ _T -> matches MERCAT)."""
    regex_pattern = pattern.replace('_', '.')  # Convert '_' to regex wildcard '.'
    matches = [word for word in df['answer'].dropna() if re.fullmatch(regex_pattern, word, re.IGNORECASE)]
    return matches

# Load trained model and vectorizer
def load_model():
    """Load the trained classifier and vectorizer."""
    global clf, vectorizer
    clf = joblib.load('crossword_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

# Example usage
load_model()
print(classify_word("Animal doctor tense after sheep rejected", "M_R_ _T"))
