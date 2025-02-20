import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import re
import pickle

class CrosswordSolver:
    def __init__(self):
        # TF-IDF for similarity matching
        self.similarity_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # TF-IDF for Random Forest
        self.rf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=1
        )
        
        # Random Forest Classifier
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42
        )
        
        self.dataset = None
        self.clue_vectors = None
        self.is_trained = False
        self.label_encoder = LabelEncoder()
        
    def preprocess_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
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
            self.dataset['processed_clues'] = self.dataset['Clue'].apply(self.preprocess_text)
            
            print("\nFirst few examples:")
            print(self.dataset[['Clue', 'Answer']].head())
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def train_model(self):
        try:
            print("\nTraining models...")
            
            # Train similarity model
            self.clue_vectors = self.similarity_vectorizer.fit_transform(self.dataset['Clue'])
            
            # Train Random Forest model
            X = self.rf_vectorizer.fit_transform(self.dataset['processed_clues'])
            y = self.label_encoder.fit_transform(self.dataset['Answer'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            
            # Calculate accuracy
            train_score = self.classifier.score(X_train, y_train)
            test_score = self.classifier.score(X_test, y_test)
            
            print(f"Random Forest Training accuracy: {train_score:.3f}")
            print(f"Random Forest Testing accuracy: {test_score:.3f}")
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def save_model(self, model_path='crossword_model.pkl'):
        try:
            model_data = {
                'similarity_vectorizer': self.similarity_vectorizer,
                'rf_vectorizer': self.rf_vectorizer,
                'classifier': self.classifier,
                'dataset': self.dataset,
                'clue_vectors': self.clue_vectors,
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
            self.similarity_vectorizer = model_data['similarity_vectorizer']
            self.rf_vectorizer = model_data['rf_vectorizer']
            self.classifier = model_data['classifier']
            self.dataset = model_data['dataset']
            self.clue_vectors = model_data['clue_vectors']
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
            
            # Preprocess the input clue
            processed_clue = self.preprocess_text(clue)
            
            predictions = []
            seen_answers = set()
            
            # 1. Try exact match first
            exact_matches = self.dataset[self.dataset['Clue'] == clue.lower()]
            if not exact_matches.empty:
                for _, row in exact_matches.iterrows():
                    answer = row['Answer']
                    if answer not in seen_answers and (not pattern or self.matches_pattern(answer, pattern)):
                        predictions.append((answer, 1.0, 'exact'))
                        seen_answers.add(answer)
            
            # 2. Try Random Forest predictions
            rf_vector = self.rf_vectorizer.transform([processed_clue])
            rf_probabilities = self.classifier.predict_proba(rf_vector)
            top_rf_indices = np.argsort(rf_probabilities[0])[-10:][::-1]
            
            for idx in top_rf_indices:
                answer = self.label_encoder.inverse_transform([idx])[0]
                probability = rf_probabilities[0][idx]
                
                if answer not in seen_answers and probability > 0.1:
                    if not pattern or self.matches_pattern(answer, pattern):
                        predictions.append((answer, probability, 'rf'))
                        seen_answers.add(answer)
            
            # 3. Try similarity-based matches
            sim_vector = self.similarity_vectorizer.transform([clue.lower()])
            similarities = cosine_similarity(sim_vector, self.clue_vectors).flatten()
            top_sim_indices = np.argsort(similarities)[-10:][::-1]
            
            for idx in top_sim_indices:
                answer = self.dataset.iloc[idx]['Answer']
                similarity = similarities[idx]
                
                if answer not in seen_answers and similarity > 0.3:
                    if not pattern or self.matches_pattern(answer, pattern):
                        predictions.append((answer, similarity, 'sim'))
                        seen_answers.add(answer)
            
            # Sort by confidence score and take top 5
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:5]
            
        except Exception as e:
            print(f"Error solving clue: {str(e)}")
            print("\nDebug info:")
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
                for i, (pred, conf, method) in enumerate(predictions, 1):
                    print(f"{i}. {pred} (confidence: {conf:.2%}, method: {method})")
            else:
                print("\nNo predictions found.")
                print("\nTry these similar clues from the dataset:")
                mask = solver.dataset['Clue'].str.contains('|'.join(clue.lower().split()))
                similar_clues = solver.dataset[mask]['Clue'].head()
                for i, similar_clue in enumerate(similar_clues, 1):
                    print(f"{i}. {similar_clue}")
                
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()