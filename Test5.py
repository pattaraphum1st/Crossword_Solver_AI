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
        # Enhanced TF-IDF parameters
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),        # Increased n-gram range
            max_features=10000,        # More features
            min_df=1,
            max_df=0.9,               # Remove too common terms
            analyzer='char_wb',        
            sublinear_tf=True,        # Add sublinear scaling
            strip_accents='unicode',
            lowercase=True
        )
        
        # Enhanced Random Forest parameters
        self.classifier = RandomForestClassifier(
            n_estimators=500,         # More trees
            max_depth=30,             # Control depth
            min_samples_split=4,      
            min_samples_leaf=2,       
            max_features='sqrt',      
            class_weight='balanced_subsample',
            bootstrap=True,
            random_state=42
        )
        
        self.dataset = None
        self.is_trained = False
        self.label_encoder = LabelEncoder()
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing with feature engineering"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Build features
        features = []
        
        # Original words
        features.extend(words)
        
        # Add word count
        features.append(f"word_count_{len(words)}")
        
        # Add first and last words
        if words:
            features.append(f"first_{words[0]}")
            features.append(f"last_{words[-1]}")
        
        # Add character count
        char_count = sum(len(word) for word in words)
        features.append(f"char_count_{char_count}")
        
        # Add word pairs
        for i in range(len(words)-1):
            features.append(f"{words[i]}_{words[i+1]}")
        
        # Add first and last letters of words
        for word in words:
            if len(word) > 0:
                features.append(f"starts_{word[0]}")
                features.append(f"ends_{word[-1]}")
        
        return ' '.join(features)
    
    def load_dataset(self, csv_path):
        try:
            self.dataset = pd.read_csv(csv_path)
            print("\nInitial Dataset Info:")
            print(f"Number of rows: {len(self.dataset)}")
            print("Columns:", self.dataset.columns.tolist())
            
            # Enhanced cleaning
            self.dataset = self.dataset.dropna()
            self.dataset['Clue'] = self.dataset['Clue'].str.lower().str.strip()
            self.dataset['Answer'] = self.dataset['Answer'].str.lower().str.strip()
            
            # Remove empty strings
            self.dataset = self.dataset[self.dataset['Clue'].str.len() > 0]
            self.dataset = self.dataset[self.dataset['Answer'].str.len() > 0]
            
            # Process clues with enhanced preprocessing
            print("\nProcessing clues...")
            self.dataset['processed_clues'] = self.dataset['Clue'].apply(self.preprocess_text)
            
            print("\nFirst few examples:")
            print(self.dataset[['Clue', 'Answer']].head())
            
            print(f"\nFinal dataset size: {len(self.dataset)} rows")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def train_model(self):
        try:
            print("\nTraining model...")
            
            # Verify data
            if self.dataset['processed_clues'].isnull().any():
                print("Error: NaN values found in processed clues")
                return False
            
            # Convert to list for vectorizer
            clues = self.dataset['processed_clues'].tolist()
            
            # Create features
            print("Creating features...")
            X = self.vectorizer.fit_transform(clues)
            y = self.label_encoder.fit_transform(self.dataset['Answer'])
            
            print(f"Feature matrix shape: {X.shape}")
            
            # Split with improved ratio
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.15,      # Smaller test set
                random_state=42,
                shuffle=True
            )
            
            # Train model
            print("Training Random Forest...")
            self.classifier.fit(X_train, y_train)
            
            # Calculate scores
            train_score = self.classifier.score(X_train, y_train)
            test_score = self.classifier.score(X_test, y_test)
            
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Testing accuracy: {test_score:.3f}")
            
            # Show feature importance
            print("\nTop important features:")
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.classifier.feature_importances_
            top_k = min(10, len(feature_names))
            top_indices = np.argsort(importances)[-top_k:]
            for idx in reversed(top_indices):
                print(f"{feature_names[idx]}: {importances[idx]:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            print("Debug info:")
            print(f"Dataset shape: {self.dataset.shape}")
            print("Sample of processed clues:")
            print(self.dataset['processed_clues'].head())
            return False
    
    def matches_pattern(self, word, pattern):
        """Enhanced pattern matching"""
        if len(word) != len(pattern):
            return False
        return all(p == '_' or w.lower() == p.lower() for w, p in zip(word, pattern))
    
    def solve(self, clue, pattern=None):
        try:
            if not self.is_trained:
                print("Error: Model has not been trained yet")
                return []
            
            # Process input clue
            processed_clue = self.preprocess_text(clue)
            
            # Check for empty clue
            if not processed_clue:
                print("Error: Empty clue after preprocessing")
                return []
            
            predictions = []
            seen_answers = set()
            
            # 1. First try exact matches
            exact_matches = self.dataset[self.dataset['Clue'] == clue.lower()]
            for _, row in exact_matches.iterrows():
                answer = row['Answer']
                if answer not in seen_answers and (not pattern or self.matches_pattern(answer, pattern)):
                    predictions.append((answer, 1.0, 'exact'))
                    seen_answers.add(answer)
            
            # 2. Then try similar clues
            similar_clues = self.dataset[self.dataset['Clue'].str.contains('|'.join(processed_clue.split()), regex=True)]
            for _, row in similar_clues.iterrows():
                answer = row['Answer']
                if answer not in seen_answers and (not pattern or self.matches_pattern(answer, pattern)):
                    predictions.append((answer, 0.8, 'similar'))
                    seen_answers.add(answer)
            
            # 3. Finally try ML predictions
            clue_vector = self.vectorizer.transform([processed_clue])
            probabilities = self.classifier.predict_proba(clue_vector)[0]
            top_indices = np.argsort(probabilities)[-20:][::-1]
            
            for idx in top_indices:
                answer = self.label_encoder.inverse_transform([idx])[0]
                prob = probabilities[idx]
                
                if answer not in seen_answers and prob > 0.05:
                    if not pattern or self.matches_pattern(answer, pattern):
                        predictions.append((answer, prob, 'ml'))
                        seen_answers.add(answer)
            
            # Sort by confidence
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:5]
            
        except Exception as e:
            print(f"Error solving clue: {str(e)}")
            print(f"Debug - Processed clue: {processed_clue}")
            return []
    
    def save_model(self, model_path='crossword_model.pkl'):
        try:
            if not self.is_trained:
                print("Error: Model has not been trained yet")
                return False
                
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
                for i, (pred, conf, method) in enumerate(predictions, 1):
                    print(f"{i}. {pred} (confidence: {conf:.2%}, method: {method})")
            else:
                print("\nNo predictions found.")
                print("Try using different words or check the pattern.")
                
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()