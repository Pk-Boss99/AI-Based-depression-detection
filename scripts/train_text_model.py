import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("data/depression.csv")
df = df[["text", "label"]]  # Adjust based on actual column names
df.dropna(inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "models/text_model.pkl")
print("✅ Model trained and saved at models/text_model.pkl")
