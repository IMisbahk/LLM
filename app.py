from flask import Flask, render_template, request, jsonify
from main import get_response, load_vocab
from models.transformer_model import TransformerModel
from data.tokenizer import SimpleTokenizer
import torch

app = Flask(__name__)

# Initialize model and tokenizer
vocab = load_vocab('data/dataset.txt')  # Your dataset path
vocab_size = len(vocab)
tokenizer = SimpleTokenizer(vocab)
model = TransformerModel(vocab_size)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if user_message:
        response = get_response(user_message, tokenizer, model)
        return jsonify({"response": response})
    return jsonify({"response": "Sorry, I didn't get that."})

if __name__ == "__main__":
    app.run(debug=True)
