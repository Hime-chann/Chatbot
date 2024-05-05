from flask import Flask, request, jsonify, render_template
import spacy
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.corpus import wordnet
import random

# Initialize Flask app
app = Flask(__name__)

# Download NLTK corpora if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Download Spacy model if not already downloaded
try:
    # Try loading the model to check if it's already downloaded
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize NLTK stopwords
stop_words = set(stopwords.words("english"))

# Download NLTK WordNet if not already downloaded
nltk.download("wordnet")

# Define common greetings
GREETINGS = ["hello", "hi", "hey", "howdy", "greetings"]

# Define possible generic responses for neutral sentiment
NEUTRAL_RESPONSES = [
    "I understand. Is there anything else I can help you with?",
    "Got it. Let me know if there's anything else I can assist you with.",
    "Thanks for letting me know. Is there anything else on your mind?",
    "Okay. If you have any other questions, feel free to ask.",
]

# Define routes
@app.route("/")
def home():
    # Render the index.html template
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_input():
    try:
        # Get user input from the request
        user_input = request.json["text"]
        
        # Preprocess text
        preprocessed_text = preprocess_text(user_input)
        
        # Tokenization
        tokens = tokenize_text(preprocessed_text)
        
        # POS tagging
        pos_tags = pos_tag_tokens(tokens)
        
        # Sentiment analysis
        sentiment = analyze_sentiment(preprocessed_text)
        
        # Process the input and generate a response
        response = generate_response(tokens, pos_tags, sentiment)
        
        # Return response to the frontend
        return jsonify({"response": response})
    
    except Exception as e:
        # Handle errors with a custom message
        return jsonify({"error": "Sorry, I can't help you with that"}), 500

def preprocess_text(text):
    # Perform text preprocessing, including removing stopwords
    text = text.lower().strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def tokenize_text(text):
    # Tokenize the text using NLTK
    tokens = nltk.word_tokenize(text)
    return tokens

def pos_tag_tokens(tokens):
    # Perform POS tagging using Spacy
    doc = nlp(" ".join(tokens))
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def analyze_sentiment(text):
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

def generate_response(tokens, pos_tags, sentiment):
    # Generate a response based on the input
    # Check if the input is a greeting
    if any(token.lower() in GREETINGS for token in tokens):
        return "Hello! How can I assist you today?"

    # Check if the input is a question
    if "?" in tokens:
        return "I'm sorry, I'm not equipped to answer questions at the moment."

    # Generate response based on sentiment
    if sentiment == "positive":
        return "I'm glad you're feeling positive!"
    elif sentiment == "negative":
        return "I'm sorry to hear that you're feeling down."
    else:
        # If sentiment is neutral, randomly select a generic response
        return random.choice(NEUTRAL_RESPONSES)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
