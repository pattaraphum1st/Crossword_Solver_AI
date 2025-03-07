import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Load the dataset
df = pd.read_csv("C:/Users/USER/Downloads/archive1/big_dave.csv")

# Drop rows where 'answer' or 'clue' is missing
df = df.dropna(subset=['answer', 'clue'])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

# Apply preprocessing to clues
df['clue'] = df['clue'].apply(preprocess_text)

# Convert clues (text) into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clue'])

# Target variable (word labels)
y = df['answer']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Function to classify a word from an input clue
def classify_word(clue, pattern):
    clue = preprocess_text(clue)  # Preprocess input clue
    clue_vector = vectorizer.transform([clue])
    predicted_word = clf.predict(clue_vector)[0]
    
    # Find matching words based on the given pattern
    possible_words = find_matching_words(pattern)
    
    if possible_words:
        if predicted_word in possible_words:
            return predicted_word
        else:
            possible_vectors = vectorizer.transform(possible_words)
            best_match_index = clf.predict_proba(clue_vector) @ possible_vectors.T
            best_word = possible_words[best_match_index.argmax()]
            return best_word
    
    return predicted_word

# Function to scope possible words based on a missing letter pattern
def find_matching_words(pattern):
    regex_pattern = pattern.replace('_', '.')  # Convert '_' to regex wildcard '.'
    matches = [word for word in df['answer'] if isinstance(word, str) and re.fullmatch(regex_pattern, word, re.IGNORECASE)]
    return matches

# Example usage
print(classify_word("Animal doctor tense after sheep rejected", "M_R_ _T"))
