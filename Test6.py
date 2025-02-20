import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import re
import pickle
import string

class CrosswordSolver:
    def __init__(self):
        # TF-IDF with adjusted parameters
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=3000,  # Reduced features
            min_df=2,          # Ignore rare terms
            max_df=0.85,       # Ignore too common terms
            analyzer='char_wb',
            strip_accents='unicode',
            lowercase=True
        )
        
        # Create multiple classifiers for ensemble
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,      # Limit depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
        
        self.gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        self.svm = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Create voting classifier
        self.classifier = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('gb', self.gb)
            ],
            voting='soft'
        )
        
        self.dataset = None
        self.is_trained = False
        self.label_encoder = LabelEncoder()

    def preprocess_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Feature engineering
        features = []
        features.extend(words)
        
        # Add word length features
        word_lengths = [len(word) for word in words]
        features.append(f"avg_len_{np.mean(word_lengths):.0f}")
        features.append(f"max_len_{max(word_lengths)}")
        
        # Add positional features
        if words:
            features.append(f"first_{words[0]}")
            features.append(f"last_{words[-1]}")
        
        return ' '.join(features)

    def load_dataset(self, csv_path):
        try:
            self.dataset = pd.read_csv(csv_path)
            print("\nInitial Dataset Info:")
            print(f"Number of rows: {len(self.dataset)}")
            print("Columns:", self.dataset.columns.tolist())
            
            # Basic cleaning
            self.dataset = self.dataset.dropna()
            self.dataset['Clue'] = self.dataset['Clue'].str.lower().str.strip()
            self.dataset['Answer'] = self.dataset['Answer'].str.lower().str.strip()
            
            # Remove duplicates
            initial_size = len(self.dataset)
            self.dataset = self.dataset.drop_duplicates(subset=['Clue', 'Answer'])
            if initial_size != len(self.dataset):
                print(f"Removed {initial_size - len(self.dataset)} duplicate entries")
            
            # Process clues
            print("\nProcessing clues...")
            self.dataset['processed_clues'] = self.dataset['Clue'].apply(self.preprocess_text)
            
            print(f"\nFinal dataset size: {len(self.dataset)} rows")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False

    def train_model(self):
        try:
            print("\nTraining model...")
            
            # Prepare features
            X = self.vectorizer.fit_transform(self.dataset['processed_clues'])
            y = self.label_encoder.fit_transform(self.dataset['Answer'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2,
                random_state=42,
                shuffle=True
            )
            
            # Perform cross-validation
            print("\nPerforming cross-validation...")
            cv_scores = cross_val_score(self.rf, X_train, y_train, cv=5)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train individual models
            print("\nTraining individual models...")
            self.rf.fit(X_train, y_train)
            self.gb.fit(X_train, y_train)
            
            # Train ensemble
            print("Training ensemble model...")
            self.classifier.fit(X_train, y_train)
            
            # Evaluate models
            print("\nModel Performance:")
            rf_train = self.rf.score(X_train, y_train)
            rf_test = self.rf.score(X_test, y_test)
            print(f"Random Forest - Train: {rf_train:.3f}, Test: {rf_test:.3f}")
            
            gb_train = self.gb.score(X_train, y_train)
            gb_test = self.gb.score(X_test, y_test)
            print(f"Gradient Boosting - Train: {gb_train:.3f}, Test: {gb_test:.3f}")
            
            ensemble_train = self.classifier.score(X_train, y_train)
            ensemble_test = self.classifier.score(X_test, y_test)
            print(f"Ensemble - Train: {ensemble_train:.3f}, Test: {ensemble_test:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False

    def solve(self, clue, pattern=None):
        try:
            if not self.is_trained:
                print("Error: Model has not been trained yet")
                return []
            
            processed_clue = self.preprocess_text(clue)
            predictions = []
            seen_answers = set()
            
            # Exact matches
            exact_matches = self.dataset[self.dataset['Clue'] == clue.lower()]
            for _, row in exact_matches.iterrows():
                answer = row['Answer']
                if answer not in seen_answers and (not pattern or self.matches_pattern(answer, pattern)):
                    predictions.append((answer, 1.0, 'exact'))
                    seen_answers.add(answer)
            
            # Similar matches
            similar_clues = self.dataset[self.dataset['Clue'].str.contains('|'.join(processed_clue.split()), regex=True)]
            for _, row in similar_clues.iterrows():
                answer = row['Answer']
                if answer not in seen_answers and (not pattern or self.matches_pattern(answer, pattern)):
                    predictions.append((answer, 0.8, 'similar'))
                    seen_answers.add(answer)
            
            # Ensemble predictions
            clue_vector = self.vectorizer.transform([processed_clue])
            
            # Get predictions from both models
            rf_proba = self.rf.predict_proba(clue_vector)[0]
            gb_proba = self.gb.predict_proba(clue_vector)[0]
            
            # Combine probabilities
            combined_proba = (rf_proba + gb_proba) / 2
            top_indices = np.argsort(combined_proba)[-10:][::-1]
            
            for idx in top_indices:
                answer = self.label_encoder.inverse_transform([idx])[0]
                prob = combined_proba[idx]
                
                if answer not in seen_answers and prob > 0.1:
                    if not pattern or self.matches_pattern(answer, pattern):
                        predictions.append((answer, prob, 'ensemble'))
                        seen_answers.add(answer)
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:5]
            
        except Exception as e:
            print(f"Error solving clue: {str(e)}")
            return []

    # [Previous methods remain the same: matches_pattern, save_model, load_model]

