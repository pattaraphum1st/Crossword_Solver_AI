import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
import pickle

class CrosswordSolver:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.dataset = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        # Handle NaN values
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    
    def load_dataset(self, csv_path):
        """Load and preprocess the dataset"""
        try:
            # Read the dataset
            self.dataset = pd.read_csv(csv_path)
            
            # Print initial dataset info
            print("\nInitial Dataset Info:")
            print(f"Number of rows: {len(self.dataset)}")
            print("Columns:", self.dataset.columns.tolist())
            
            # Check for required columns
            if 'Clue' not in self.dataset.columns or 'Answer' not in self.dataset.columns:
                raise ValueError("CSV file must contain 'Clue' and 'Answer' columns")
            
            # Clean the dataset
            print("\nCleaning dataset...")
            
            # Remove rows with NaN values
            initial_rows = len(self.dataset)
            self.dataset = self.dataset.dropna()
            rows_after_dropna = len(self.dataset)
            if initial_rows != rows_after_dropna:
                print(f"Removed {initial_rows - rows_after_dropna} rows with missing values")
            
            # Remove rows where Clue or Answer is empty string after stripping whitespace
            self.dataset = self.dataset[self.dataset['Clue'].str.strip().astype(bool)]
            self.dataset = self.dataset[self.dataset['Answer'].str.strip().astype(bool)]
            rows_after_empty = len(self.dataset)
            if rows_after_dropna != rows_after_empty:
                print(f"Removed {rows_after_dropna - rows_after_empty} rows with empty strings")
            
            # Create processed clues column
            self.dataset['processed_clues'] = self.dataset['Clue'].apply(self.preprocess_text)
            
            print("\nFinal Dataset Info:")
            print(f"Final number of rows: {len(self.dataset)}")
            print("\nFirst few rows of the cleaned dataset:")
            print(self.dataset[['Clue', 'Answer']].head())
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Could not find the file at {csv_path}")
            return False
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty")
            return False
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def train_model(self):
        """Train the model using TF-IDF and Random Forest"""
        try:
            if self.dataset is None:
                raise ValueError("No dataset loaded. Please load a dataset first.")
            
            if len(self.dataset) == 0:
                raise ValueError("Dataset is empty after cleaning")
            
            print("\nStarting model training...")
            print(f"Training with {len(self.dataset)} examples...")
                
            X = self.vectorizer.fit_transform(self.dataset['processed_clues'])
            y = self.dataset['Answer']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            
            # Print accuracy score
            score = self.classifier.score(X_test, y_test)
            print(f"Model accuracy: {score:.2f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            print("Debugging info:")
            print(f"Dataset shape: {self.dataset.shape}")
            print("Sample of processed clues:")
            print(self.dataset['processed_clues'].head())
            return False
        
    def save_model(self, model_path='crossword_model.pkl'):
        """Save the trained model and vectorizer"""
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")
                
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
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
        """Load a trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.is_trained = model_data.get('is_trained', True)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def matches_pattern(self, word, pattern):
        """Check if a word matches the given pattern (e.g., 'd_g')"""
        if len(word) != len(pattern):
            return False
        
        for w_char, p_char in zip(word, pattern):
            if p_char != '_' and p_char.lower() != w_char.lower():
                return False
        return True
    
    def solve(self, clue, pattern=None):
        """Predict answer based on clue and optional pattern"""
        try:
            if not self.is_trained:
                print("Error: Model has not been trained yet")
                return []
                
            # Preprocess the clue
            processed_clue = self.preprocess_text(clue)
            
            # Transform the clue using the trained vectorizer
            clue_vector = self.vectorizer.transform([processed_clue])
            
            # Get top 5 predictions
            probabilities = self.classifier.predict_proba(clue_vector)
            top_indices = np.argsort(probabilities[0])[-5:][::-1]
            predictions = [self.classifier.classes_[i] for i in top_indices]
            
            # Filter by pattern if provided
            if pattern:
                predictions = [pred for pred in predictions if self.matches_pattern(pred, pattern)]
            
            return predictions
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