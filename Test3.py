import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle

class CrosswordSolver:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True
        )
        self.dataset = None
        self.clue_vectors = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        # Simple preprocessing
        text = str(text).lower().strip()
        return text
    
    def load_dataset(self, csv_path):
        try:
            self.dataset = pd.read_csv(csv_path)
            print("\nInitial Dataset Info:")
            print(f"Number of rows: {len(self.dataset)}")
            print("Columns:", self.dataset.columns.tolist())
            
            # Basic cleaning
            self.dataset = self.dataset.dropna()
            self.dataset['Clue'] = self.dataset['Clue'].str.lower()
            self.dataset['Answer'] = self.dataset['Answer'].str.lower()
            
            print("\nFirst few examples:")
            print(self.dataset.head())
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def train_model(self):
        try:
            print("\nTraining model...")
            
            # Create TF-IDF vectors for all clues
            self.clue_vectors = self.vectorizer.fit_transform(self.dataset['Clue'])
            print("Model trained successfully!")
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def save_model(self, model_path='crossword_model.pkl'):
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'dataset': self.dataset,
                'clue_vectors': self.clue_vectors,
                'is_trained': self.is_trained
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path='crossword_model.pkl'):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.dataset = model_data['dataset']
            self.clue_vectors = model_data['clue_vectors']
            self.is_trained = model_data.get('is_trained', True)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def matches_pattern(self, word, pattern):
        if len(word) != len(pattern):
            return False
        for w_char, p_char in zip(word, pattern):
            if p_char != '_' and p_char.lower() != w_char.lower():
                return False
        return True
    
    def solve(self, clue, pattern=None):
        try:
            if not self.is_trained:
                print("Error: Model has not been trained yet")
                return []
            
            # Preprocess the input clue
            processed_clue = self.preprocess_text(clue)
            
            # First try exact match
            exact_matches = self.dataset[self.dataset['Clue'] == processed_clue]
            if not exact_matches.empty:
                predictions = exact_matches['Answer'].tolist()
                if pattern:
                    predictions = [pred for pred in predictions if self.matches_pattern(pred, pattern)]
                if predictions:
                    return [(pred, 1.0) for pred in predictions[:5]]
            
            # Transform the input clue
            clue_vector = self.vectorizer.transform([processed_clue])
            
            # Calculate similarities with all clues
            similarities = cosine_similarity(clue_vector, self.clue_vectors).flatten()
            
            # Get top 10 most similar clues
            top_indices = np.argsort(similarities)[-10:][::-1]
            
            predictions = []
            seen_answers = set()
            
            for idx in top_indices:
                answer = self.dataset.iloc[idx]['Answer']
                similarity = similarities[idx]
                
                # Skip if we've seen this answer or similarity is too low
                if answer in seen_answers or similarity < 0.1:
                    continue
                
                if not pattern or self.matches_pattern(answer, pattern):
                    predictions.append((answer, similarity))
                    seen_answers.add(answer)
            
            return predictions[:5]
            
        except Exception as e:
            print(f"Error solving clue: {str(e)}")
            print("Debug info:")
            print(f"Input clue: {clue}")
            print(f"Processed clue: {processed_clue}")
            print(f"Pattern: {pattern}")
            return []

def main():
    solver = CrosswordSolver()
    
    while True:
        print("\nCrossword Solver Menu:")
        print("1. Load dataset and train new model")
        print("2. Load existing model")
        print("3. Solve clue")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            csv_path = input("Enter the path to your CSV dataset: ")
            if solver.load_dataset(csv_path):
                if solver.train_model():
                    if solver.save_model():
                        print("Model trained and saved successfully!")
                
        elif choice == '2':
            model_path = input("Enter the path to your model file (default: crossword_model.pkl): ") or 'crossword_model.pkl'
            if solver.load_model(model_path):
                print("Model loaded successfully!")
                
        elif choice == '3':
            if not solver.is_trained:
                print("Please train or load a model first!")
                continue
                
            clue = input("Enter the clue: ")
            pattern = input("Enter the pattern (use underscores for blanks, e.g., 'd_g') or press Enter to skip: ")
            
            predictions = solver.solve(clue, pattern if pattern else None)
            
            if predictions:
                print("\nTop predictions:")
                for i, (pred, prob) in enumerate(predictions, 1):
                    print(f"{i}. {pred} (similarity: {prob:.2%})")
            else:
                print("\nNo predictions found.")
                print("Debug info - try these variations:")
                # Show similar clues from dataset
                mask = solver.dataset['Clue'].str.contains('|'.join(clue.lower().split()))
                similar_clues = solver.dataset[mask]['Clue'].head()
                print("\nSimilar clues in dataset:")
                for i, similar_clue in enumerate(similar_clues, 1):
                    print(f"{i}. {similar_clue}")
                
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()