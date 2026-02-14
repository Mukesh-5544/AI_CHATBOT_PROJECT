import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents
with open(r"C:\Users\91970\Downloads\intents.json") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

# Train model
model = MultinomialNB()
model.fit(X, tags)

print("🤖 ML Chatbot ready! (type 'bye' to exit)")

# Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break

    user_X = vectorizer.transform([user_input])
    prediction = model.predict(user_X)[0]

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            print("Bot:", random.choice(intent["responses"]))
            break
