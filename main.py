import torch
from models.transformer_model import TransformerModel
from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset  # Import from text_dataset.py
from training.train import train  # Import from train.py

# Load vocabulary from the dataset
def load_vocab(file_path):
    vocab = ['<pad>', '<unk>']  # Start with pad and unknown tokens
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if word not in vocab:
                    vocab.append(word)
    return vocab

# Define response generation function
def get_response(message, tokenizer, model):
    input_tokens = tokenizer.tokenize(message)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)  # Pass the tensor to the model

        if output.dim() == 3:
            probabilities = torch.softmax(output[:, -1, :], dim=-1)  # Get probabilities for the last token
        elif output.dim() == 2:
            probabilities = torch.softmax(output, dim=-1)  # Already in 2D
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

    if probabilities.dim() != 2:
        raise ValueError(f"Probabilities must be 1 or 2D but got {probabilities.dim()}D")

    predicted_token = torch.multinomial(probabilities, num_samples=1).item()
    return tokenizer.idx_to_vocab(predicted_token)

if __name__ == "__main__":
    vocab = load_vocab('data/dataset.txt')  # Path to your dataset
    vocab_size = len(vocab)
    tokenizer = SimpleTokenizer(vocab)
    model = TransformerModel(vocab_size)

    # Set the model to evaluation mode for inference
    model.eval()

    # Test response
    input_text = "hello world"
    response = get_response(input_text, tokenizer, model)
    print("Input:", input_text)
    print("Response:", response)

    # Initialize dataset and train model
    dataset = TextDataset('data/dataset.txt', tokenizer)  # Provide your dataset path
    train(model, dataset, epochs=50, batch_size=32)
