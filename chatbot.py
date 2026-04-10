"""
====================================================
  AI CHATBOT WITH NATURAL LANGUAGE PROCESSING
  Built with NLTK | Python 3.x
====================================================
"""

import nltk
import random
import string
import warnings
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings("ignore")

# ── Download required NLTK data ──────────────────────────────────────────────
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_data()

# ── Knowledge Base ────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans.
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
Machine learning is a branch of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed.
Deep learning is a subset of machine learning that uses neural networks with many layers to learn complex patterns from large amounts of data.
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.
NLTK stands for Natural Language Toolkit and is a leading platform for building Python programs to work with human language data.
Python is a high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991.
Python is known for its readability and simplicity. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
A chatbot is a software application designed to simulate human conversation through text or voice interactions.
Chatbots can be rule-based or AI-powered. Rule-based chatbots follow predefined scripts, while AI-powered chatbots use machine learning and NLP.
Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge from structured and unstructured data.
Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.
Computer vision is a field of AI that enables computers to interpret and understand visual information from the world.
Robotics is the intersection of engineering, computer science, and AI that deals with the design, construction, and operation of robots.
Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power.
The internet is a global system of interconnected computer networks that uses the TCP/IP protocol suite to communicate.
Cybersecurity involves protecting computer systems and networks from theft of or damage to hardware, software, or data.
Big data refers to data sets that are so large or complex that traditional data processing applications are inadequate.
Blockchain is a distributed ledger technology that records transactions across multiple computers so that records cannot be altered.
The Internet of Things (IoT) refers to physical objects with sensors, processing ability, and software that connect and exchange data with other devices.
"""

# ── Predefined Q&A Pairs ──────────────────────────────────────────────────────
QA_PAIRS = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "good morning", "good afternoon",
                     "good evening", "howdy", "greetings", "what's up", "sup"],
        "responses": [
            "Hello! 👋 I'm your AI assistant. How can I help you today?",
            "Hi there! Great to see you. What would you like to know?",
            "Hey! I'm ready to help. What's on your mind?",
        ]
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "later", "take care", "quit",
                     "exit", "farewell", "ciao", "adios"],
        "responses": [
            "Goodbye! 👋 Have a wonderful day!",
            "See you later! Come back anytime you have questions.",
            "Take care! It was great chatting with you.",
        ]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "thank", "appreciate", "grateful",
                     "cheers", "thx", "ty"],
        "responses": [
            "You're welcome! 😊 Is there anything else I can help you with?",
            "Happy to help! Don't hesitate to ask more questions.",
            "My pleasure! Let me know if you need anything else.",
        ]
    },
    "how_are_you": {
        "patterns": ["how are you", "how do you do", "how is it going",
                     "how are things", "you okay", "are you okay"],
        "responses": [
            "I'm doing great, thank you for asking! 🤖 Ready to answer your questions.",
            "I'm functioning perfectly! How about you?",
            "All systems operational! What can I help you with?",
        ]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "what are you called",
                     "your name", "tell me your name"],
        "responses": [
            "I'm NLPBot 🤖 — your AI-powered chatbot built with NLTK and Python!",
            "You can call me NLPBot! I'm an AI chatbot powered by Natural Language Processing.",
        ]
    },
    "capabilities": {
        "patterns": ["what can you do", "your capabilities", "what do you know",
                     "help me", "features", "abilities", "skills"],
        "responses": [
            "I can answer questions about:\n  🧠 Artificial Intelligence & Machine Learning\n  🐍 Python Programming\n  📊 Data Science & Big Data\n  💬 NLP & Chatbots\n  🌐 Technology Topics\n\nJust ask me anything!",
        ]
    },
    "jokes": {
        "patterns": ["tell me a joke", "joke", "funny", "make me laugh", "humor"],
        "responses": [
            "Why do programmers prefer dark mode? 🌙 Because light attracts bugs! 😄",
            "Why did the neural network go to therapy? It had too many deep issues! 🤣",
            "What do you call a programmer from Finland? Nerdic! 😂",
            "Why was the computer cold? It left its Windows open! 🥶",
        ]
    },
}

# ── NLP Utility Functions ──────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Tokenize, lowercase, remove stopwords & punctuation, then lemmatize."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def get_pos_tags(text: str) -> list:
    """Return POS tags for a given sentence."""
    tokens = word_tokenize(text)
    return pos_tag(tokens)

def get_intent(user_input: str) -> tuple[str | None, str | None]:
    """Match user input against QA_PAIRS patterns."""
    text = user_input.lower().strip()
    for intent, data in QA_PAIRS.items():
        for pattern in data["patterns"]:
            if pattern in text:
                return intent, random.choice(data["responses"])
    return None, None

def get_knowledge_response(user_input: str) -> str:
    """TF-IDF cosine similarity against the knowledge base."""
    sentences = sent_tokenize(KNOWLEDGE_BASE)
    sentences_clean = [preprocess_text(s) for s in sentences]
    user_clean = preprocess_text(user_input)

    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform(sentences_clean + [user_clean])
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        best_idx = int(np.argmax(similarity))
        best_score = float(similarity[0][best_idx])

        if best_score > 0.15:
            return f"📚 {sentences[best_idx]}"
        else:
            return ("🤔 I'm not sure about that. You could try rephrasing, or ask about "
                    "AI, Machine Learning, Python, NLP, Data Science, or Technology topics!")
    except Exception:
        return "⚠️  Sorry, I had trouble processing that. Please try again."

# ── Sentiment Detector (lexicon-based) ───────────────────────────────────────
POSITIVE_WORDS = {"good", "great", "awesome", "excellent", "amazing", "love",
                  "happy", "wonderful", "fantastic", "nice", "best", "brilliant"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "horrible", "worst",
                  "sad", "angry", "disappointing", "poor", "ugly", "broken"}

def detect_sentiment(text: str) -> str:
    tokens = set(word_tokenize(text.lower()))
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    if pos > neg:
        return "😊 Positive"
    elif neg > pos:
        return "😞 Negative"
    return "😐 Neutral"

# ── Main Chat Engine ──────────────────────────────────────────────────────────
class NLPChatBot:
    def __init__(self):
        self.name = "NLPBot"
        self.conversation_history: list[dict] = []

    def respond(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please type something! I'm listening... 👂"

        # Store history
        self.conversation_history.append({"role": "user", "text": user_input})

        # 1. Intent matching
        intent, response = get_intent(user_input)
        if response:
            self.conversation_history.append({"role": "bot", "text": response})
            return response

        # 2. Sentiment query
        if any(w in user_input.lower() for w in ["sentiment", "feeling", "emotion", "mood"]):
            sent = detect_sentiment(user_input)
            response = f"Detected sentiment in your message: {sent}"
            self.conversation_history.append({"role": "bot", "text": response})
            return response

        # 3. POS tagging request
        if any(w in user_input.lower() for w in ["pos tag", "parts of speech", "tag this", "analyze"]):
            tags = get_pos_tags(user_input)
            tag_str = ", ".join([f"{w}({t})" for w, t in tags[:8]])
            response = f"🏷️  POS Tags: {tag_str}"
            self.conversation_history.append({"role": "bot", "text": response})
            return response

        # 4. Knowledge base similarity search
        response = get_knowledge_response(user_input)
        self.conversation_history.append({"role": "bot", "text": response})
        return response

    def run(self):
        """Run chatbot in the terminal."""
        print("\n" + "="*55)
        print("       🤖  NLPBot — AI Chatbot with NLTK NLP")
        print("="*55)
        print(" Type your message and press Enter to chat.")
        print(" Type 'quit' or 'exit' to stop the bot.")
        print(" Type 'history' to see conversation history.")
        print(" Type 'sentiment <text>' to detect sentiment.")
        print("="*55 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
                    print("NLPBot: Goodbye! 👋 Have a great day!\n")
                    break
                if user_input.lower() == "history":
                    print("\n── Conversation History ──")
                    for msg in self.conversation_history:
                        prefix = "You" if msg["role"] == "user" else "Bot"
                        print(f"  {prefix}: {msg['text']}")
                    print()
                    continue
                response = self.respond(user_input)
                print(f"NLPBot: {response}\n")
            except KeyboardInterrupt:
                print("\nNLPBot: Goodbye! 👋\n")
                break

# ── Demo Run (shows sample outputs without user input) ───────────────────────
def run_demo():
    bot = NLPChatBot()
    demo_inputs = [
        "Hello!",
        "What is your name?",
        "What can you do?",
        "Tell me about machine learning",
        "What is Python?",
        "What is NLP?",
        "What is deep learning?",
        "Tell me a joke",
        "Thank you",
        "Goodbye"
    ]

    print("\n" + "="*60)
    print("       🤖  NLPBot — DEMO OUTPUT (Sample Conversations)")
    print("="*60 + "\n")

    for query in demo_inputs:
        response = bot.respond(query)
        print(f"  👤 User   : {query}")
        print(f"  🤖 NLPBot : {response}")
        print()

    print("="*60)
    print("  ✅ Demo complete! Run bot.run() for interactive mode.")
    print("="*60 + "\n")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    bot = NLPChatBot()
    if "--demo" in sys.argv:
        run_demo()
    else:
        # Show demo first, then enter interactive mode
        run_demo()
        print("\n🚀 Entering Interactive Mode...\n")
        bot.run()