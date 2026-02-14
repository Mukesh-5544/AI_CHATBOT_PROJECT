import nltk
import string
from nltk.stem import WordNetLemmatizer

lemmer = WordNetLemmatizer()

knowledge = """
Artificial intelligence is the simulation of human intelligence by machines.
AI is used in chatbots, self-driving cars, and recommendation systems.
Machine learning is a subset of AI.
Python is a popular language for AI.
Deep learning is a part of machine learning that uses neural networks.
"""

sentences = nltk.sent_tokenize(knowledge)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return [lemmer.lemmatize(word) for word in tokens if word not in string.punctuation]

def chatbot_response(user_input):
    user_tokens = preprocess(user_input)

    best_match = None
    best_score = 0

    for sentence in sentences:
        sentence_tokens = preprocess(sentence)

        common_words = set(user_tokens).intersection(set(sentence_tokens))
        score = len(common_words)

        if score > best_score:
            best_score = score
            best_match = sentence

    if best_score == 0:
        return "Sorry, I don't understand."
    else:
        return best_match

print("🤖 NLP Chatbot: Ask me about AI (type 'bye' to exit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break
    else:
        print("Bot:", chatbot_response(user_input))
