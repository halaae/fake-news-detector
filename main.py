import pandas as pd

# Load CSVs from dataset folder
fake = pd.read_csv('dataset/Fake.csv')
true = pd.read_csv('dataset/True.csv')

# Label the data: 0 = fake, 1 = true
fake['label'] = 0
true['label'] = 1

# Combine both into one dataset
df = pd.concat([fake, true])
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Preview
print("Dataset loaded successfully ✅")
print(df.head())
print("\nLabel counts:\n", df['label'].value_counts())

import string
import re

# Combine title + text into one column
df['text'] = df['title'] + " " + df['text']

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    return text

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

print("\n✅ Text cleaning complete")
print(df['text'].head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Separate features and labels
X = df['text']
y = df['label']

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n✅ Data split complete")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n✅ TF-IDF vectorization complete")
print("TF-IDF shape:", X_train_tfidf.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Detailed report
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save model and vectorizer
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved in 'models/' folder")




