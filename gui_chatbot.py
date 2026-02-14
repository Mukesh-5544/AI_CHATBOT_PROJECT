def chatbot_response(text):
    X = vectorizer.transform([text])

    probs = model.predict_proba(X)
    confidence = probs.max()

    tag = model.predict(X)[0]

    if confidence < 0.4:
        return "I am not sure. Can you rephrase?"

    for intent in data["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return f"{response} (confidence: {confidence:.2f})"

    return "Sorry, I don't understand."
