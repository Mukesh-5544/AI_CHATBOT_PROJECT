from flask import Flask, render_template, request, jsonify
import pickle
import json
import random

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    data = json.load(file)

def chatbot_response(text):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)
    confidence = probs.max()
    tag = model.predict(X)[0]

    if confidence < 0.4:
        return "I am not sure. Please ask something else."

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    user_input = request.form["msg"]
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
