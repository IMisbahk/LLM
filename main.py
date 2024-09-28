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
def get_response(input_text):
    input_tokens = tokenizer.tokenize(input_text).unsqueeze(0)  # Add batch dimension
    output = model(input_tokens)
    predicted_tokens = output.argmax(dim=-1).squeeze(0)  # Get the highest probability tokens
    return tokenizer.detokenize(predicted_tokens.tolist())

if __name__ == "__main__":
    # Example usage
    print(get_response("hello world this is a test"))
