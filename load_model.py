import re
from joblib import load
import numpy as np

model = load("naive_bayes_crossword.pkl", mmap_mode="r")
vectorizer = load("vectorizer.pkl")

def preprocess_text(text):
    """Cleans the clue text by removing unnecessary characters and lowercasing."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text) 
    return text

def match_pattern(word, pattern):
    """Checks if a word strictly matches the pattern, considering known letters."""
    if len(word) != len(pattern):
        return False

    for w_char, p_char in zip(word, pattern):
        if p_char != "_" and w_char != p_char:
            return False
    return True 

def predict_word(clue, pattern, top_n=100):
    """Predicts crossword words strictly matching the pattern."""
    
    clue = preprocess_text(clue)
    X_input = vectorizer.transform([clue])

    predicted_probs = model.predict_proba(X_input)

    class_labels = model.classes_  
    top_indices = np.argsort(predicted_probs[0])[::-1][:top_n]  
    predictions = [class_labels[i] for i in top_indices] 

    word_length = len(pattern)  
    first_letter = pattern[0].lower() if pattern[0] != "_" else None 

    filtered_predictions = [
        word for word in predictions
        if len(word) == word_length and match_pattern(word, pattern)
    ]

    if first_letter:
        filtered_predictions = [word for word in filtered_predictions if word.startswith(first_letter)]

    if not filtered_predictions:
        filtered_predictions = [word for word in predictions if len(word) == word_length]

    return filtered_predictions if filtered_predictions else ["No matching word found"]

while True:
    clue = input("\nEnter crossword clue (or type 'exit' to quit): ")
    if clue.lower() == "exit":
        break

    pattern = input("Enter pattern (use _ for missing letters): ").strip()
    predicted_words = predict_word(clue, pattern)

    print(f"\nwords: {', '.join(predicted_words)}")
