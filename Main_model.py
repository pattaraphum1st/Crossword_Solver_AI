import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Optimize Dataset
def optimize_csv(file_path, output_path):
    print("🔍 Optimizing dataset...")
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    expected_columns = ["Clue", "Word"]
    if not all(col in df.columns for col in expected_columns):
        raise KeyError(f"❌ Missing required columns {expected_columns}. Check CSV structure.")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=["Clue", "Word"], inplace=True)

    df['Clue'] = df['Clue'].astype(str).str.lower().str.strip()
    df['Word'] = df['Word'].astype(str).str.lower().str.strip()

    df.to_csv(output_path, index=False, encoding="ISO-8859-1")
    print(f"✅ Optimized CSV saved at: {output_path}")
    return output_path

# ✅ Preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}  # Keep negations

    if not isinstance(text, str) or text.strip() == "":
        return "unknownclue"

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words).strip() if words else "unknownclue"

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df['Clue'] = df['Clue'].astype(str).map(preprocess_text)
    return df

# ✅ Preprocess & Balance Data (Memory Efficient)
def preprocess_data(df):
    df_balanced = df.groupby("Word", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 100), random_state=42))
    df_balanced = df_balanced.reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=3000, dtype=np.float32)  # ✅ Lowered features (Saves memory)
    X = vectorizer.fit_transform(df_balanced['Clue'])
    y = df_balanced['Word']

    print(f"✅ Data Size: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, vectorizer

# ✅ Train Naive Bayes in Batches (Memory Efficient)
def train_model(X, y):
    print("\n🛠 Training Naive Bayes Model... (Using Batch Training)")

    model = MultinomialNB(alpha=0.1)  # 🚀 Smoothing to prevent zero probabilities

    batch_size = 25000  # ✅ Smaller batch to reduce RAM usage
    num_batches = X.shape[0] // batch_size + 1

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, X.shape[0])
        
        if start >= end:
            break
        
        print(f"\n🚀 Training Batch {i+1}/{num_batches} ({start} to {end})")
        model.partial_fit(X[start:end], y[start:end], classes=np.unique(y)) 

        gc.collect()

    print("✅ Training Completed!")
    return model

def evaluate_model(model, X_test, y_test):
    print("\n📊 Evaluating model...")

    batch_size = 5000  
    y_pred = []

    for i in range(0, X_test.shape[0], batch_size):
        print(f"🚀 Evaluating Batch {i // batch_size + 1}")
        batch_pred = model.predict(X_test[i:i + batch_size])
        y_pred.extend(batch_pred)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")

    print("\n📌 Classification Report (Showing Top 50 Words):\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n📊 Plotting Top 50 Class Distribution Instead...")
    
    most_common_words = [word for word, _ in Counter(y_test).most_common(50)]

    y_test_filtered = [word if word in most_common_words else "OTHER" for word in y_test]
    y_pred_filtered = [word if word in most_common_words else "OTHER" for word in y_pred]

    conf_matrix = confusion_matrix(y_test_filtered, y_pred_filtered, labels=most_common_words + ["OTHER"])

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=False, cmap="Blues", fmt="d", xticklabels=most_common_words + ["OTHER"], yticklabels=most_common_words + ["OTHER"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Top 50 Classes + Other)")
    plt.xticks(rotation=90)
    plt.show()

def main(file_path):
    try:
        print("🚀 Starting Crossword Solver Training...")

        optimized_path = "optimized_crossword_dataset.csv"
        file_path = optimize_csv(file_path, optimized_path)

        df = load_data(file_path)

        print("\n📊 Preprocessing Data...")
        X, y, vectorizer = preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\n🛠 Training model...")
        model = train_model(X_train, y_train)

        dump(model, "naive_bayes_crossword1.pkl")
        dump(vectorizer, "vectorizer1.pkl")
        print("\n✅ Model and vectorizer saved successfully!")

        evaluate_model(model, X_test, y_test)

    except Exception as e:
        print(f"❌ Script crashed with error: {e}")

if __name__ == "__main__":
    dataset_path = "C:/Users/USER/Downloads/KaggleNYT/nytcrosswords.csv"
    main(dataset_path)
