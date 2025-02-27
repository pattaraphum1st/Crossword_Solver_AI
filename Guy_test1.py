import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:/Users/USER/Downloads/crossword_dataset3 - Sheet1.csv")

# Drop rows where 'Word' is missing
df = df.dropna(subset=['Word'])

# Convert properties (text) into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Properties'])

# Target variable (word labels)
y = df['Word']

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
def classify_word(clue):
    clue_vector = vectorizer.transform([clue])
    prediction = clf.predict(clue_vector)
    return prediction[0]

# Function to scope possible words based on a missing letter pattern
def find_matching_words(pattern):
    regex_pattern = pattern.replace('_', '.')  # Convert '_' to regex wildcard '.'
    matches = [word for word in df['Word'] if re.fullmatch(regex_pattern, word)]
    return matches

# Example usage
print(classify_word("precious element used in jewelry"))  # Predict word from clue

