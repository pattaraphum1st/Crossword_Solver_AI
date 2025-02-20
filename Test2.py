import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import pickle
import string

class CrosswordSolver:
    def __init__(self):
        # Improved TF-IDF parameters
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            stop_words='english'
        )
        # Improved Random Forest parameters
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        )
        self.dataset = None
        self.is_trained = False
        self.label_encoder = LabelEncoder()
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        text = ' '.join(text.split())
        text = f"{text} length_{len(text.split())}"
        return text
    
    def augment_data(self, clue, answer):
        variations = [clue]
        word_count = len(clue.split())
        variations.append(f"{clue} words_{word_count}")
        variations.append(f"{clue} chars_{len(answer)}")
        if len(answer) > 0:
            variations.append(f"{clue} starts_{answer[0]}")
        return variations
    
    def load_dataset(self, csv_path):
        try:
            self.dataset = pd.read_csv(csv_path)
            
            print("\nInitial Dataset Info:")
            print(f"Number of rows: {len(self.dataset)}")
            print("Columns:", self.dataset.columns.tolist())
            
            if 'Clue' not in self.dataset.columns or 'Answer' not in self.dataset.columns:
                raise ValueError("CSV file must contain 'Clue' and 'Answer' columns")
            
            print("\nCleaning dataset...")
            
            # Remove rows with NaN values
            initial_rows = len(self.dataset)
            self.dataset = self.dataset.dropna()
            rows_after_dropna = len(self.dataset)
            
            # Remove rows with empty strings
            self.dataset = self.dataset[self.dataset['Clue'].str.strip().astype(bool)]
            self.dataset = self.dataset[self.dataset['Answer'].str.strip().astype(bool)]
            
            # Convert answers to lowercase
            self.dataset['Answer'] = self.dataset['Answer'].str.lower()
            
            # Create processed clues
            self.dataset['processed_clues'] = self.dataset['Clue'].apply(self.preprocess_text)
            
            # Encode the answers
            self.label_encoder.fit(self.dataset['Answer'])
            
            print(f"\nFinal dataset size: {len(self.dataset)} rows")
            print("\nSample of processed data:")
            print(self.dataset[['Clue', 'Answer', 'processed_clues']].head())
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def train_model(self):
        try:
            if self.dataset is None or len(self.dataset) == 0:
                raise ValueError("No valid dataset loaded")
            
            print("\nStarting model training...")
            print(f"Training with {len(self.dataset)} examples...")
            
            # Transform text features
            X = self.vectorizer.fit_transform(self.dataset['processed_clues'])
            
            # Transform labels
            y = self.label_encoder.transform(self.dataset['Answer'])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.classifier.fit(X_train, y_train)
            
            # Calculate accuracy
            train_score = self.classifier.score(X_train, y_train)
            test_score = self.classifier.score(X_test, y_test)
            
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Testing accuracy: {test_score:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def save_model(self, model_path='crossword_model.pkl'):
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")
            
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
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
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
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
            
            # Preprocess the clue
            processed_clue = self.preprocess_text(clue)
            
            # Transform the clue
            clue_vector = self.vectorizer.transform([processed_clue])
            
            # Get predictions
            probabilities = self.classifier.predict_proba(clue_vector)
            top_indices = np.argsort(probabilities[0])[-10:][::-1]
            predictions = [self.label_encoder.inverse_transform([i])[0] for i in top_indices]
            
            # Filter by pattern if provided
            if pattern:
                predictions = [pred for pred in predictions if self.matches_pattern(pred, pattern)]
            
            return predictions[:5]
            
        except Exception as e:
            print(f"Error solving clue: {str(e)}")
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
                for i, pred in enumerate(predictions, 1):
                    print(f"{i}. {pred}")
            else:
                print("No predictions found.")
                
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()