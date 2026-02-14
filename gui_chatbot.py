import json
import random
import tkinter as tk
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

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

model = MultinomialNB()
model.fit(X, tags)

# Chatbot response function
def get_response(user_input):
    user_X = vectorizer.transform([user_input])
    prediction = model.predict(user_X)[0]

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

# GUI setup
window = tk.Tk()
window.title("AI Chatbot")

chat_area = tk.Text(window, height=20, width=50)
chat_area.pack()

entry_box = tk.Entry(window, width=40)
entry_box.pack()

def send():
    user_input = entry_box.get()
    chat_area.insert(tk.END, "You: " + user_input + "\n")

    if user_input.lower() == "bye":
        chat_area.insert(tk.END, "Bot: Goodbye!\n")
        window.quit()
    else:
        response = get_response(user_input)
        chat_area.insert(tk.END, "Bot: " + response + "\n")

    entry_box.delete(0, tk.END)

send_button = tk.Button(window, text="Send", command=send)
send_button.pack()

window.mainloop()

