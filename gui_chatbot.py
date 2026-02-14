import tkinter as tk
import pickle
import json
import random

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents
with open("intents.json") as file:
    data = json.load(file)

def chatbot_response(text):
    X = vectorizer.transform([text])
    tag = model.predict(X)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."

# Send message
def send():
    user_input = entry.get()
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_input + "\n")

    response = chatbot_response(user_input)
    chat_log.insert(tk.END, "Bot: " + response + "\n\n")

    entry.delete(0, tk.END)
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

# GUI setup
window = tk.Tk()
window.title("AI Chatbot 🤖")
window.geometry("500x600")

chat_log = tk.Text(window, bd=1, bg="white", font=("Arial", 12))
chat_log.config(state=tk.DISABLED)
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(window, bd=1, font=("Arial", 12))
entry.pack(padx=10, pady=10, fill=tk.X)

send_button = tk.Button(window, text="Send", command=send)
send_button.pack(pady=5)

window.mainloop()

