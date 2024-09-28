# main.py
import torch
from models.transformer_model import TransformerModel
from utils.tokenizer import SimpleTokenizer

# Load or define vocabulary
vocab = ['<pad>', '<unk>', 'hello', 'world', 'this', 'is', 'a', 'test']
vocab_size = len(vocab)

# Initialize tokenizer and model
tokenizer = SimpleTokenizer(vocab)
model = TransformerModel(vocab_size)

# Sample input text
input_text = "hello world this is a test"

# Tokenize input
input_tokens = tokenizer.tokenize(input_text).unsqueeze(0)  # Add batch dimension

# Run the model
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(input_tokens)

# Print output (convert tokens back to words)
predicted_tokens = output.argmax(dim=-1).squeeze(0)  # Get the highest probability tokens
output_text = tokenizer.detokenize(predicted_tokens.tolist())

print("Input:", input_text)
print("Output:", output_text)
