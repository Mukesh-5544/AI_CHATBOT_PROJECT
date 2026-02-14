import json
import random
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Check if model already exists
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    print("✅ Loading trained model...")
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
else:
    print("⚡ Training model...")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(patterns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, tags, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"🎯 Model Accuracy: {acc * 100:.2f}%")

    # Save model
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Chat function
def chatbot_response(text):
    X = vectorizer.transform([text])
    tag = model.predict(X)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."

# Chat loop
print("🤖 ML Chatbot ready! (type 'bye' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("Bot:", response)
