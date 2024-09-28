import json
import torch
from models.transformer_model import TransformerModel
from utils.tokenizer import SimpleTokenizer
import torch.optim as optim
import re
from torch.utils.data import Dataset, DataLoader

# Load or define vocabulary
vocab = ['<pad>', '<unk>', 'hello', 'world', 'this', 'is', 'a', 'test']
vocab_size = len(vocab)

# Initialize tokenizer and model
tokenizer = SimpleTokenizer(vocab)
model = TransformerModel(vocab_size)

# Set the model to evaluation mode
model.eval()

def get_response(message):
    # Tokenize the input message
    input_tokens = tokenizer.tokenize(message).unsqueeze(0)  # Add batch dimension
    output = model(input_tokens)

    # Get the predicted tokens
    predicted_tokens = output.argmax(dim=-1).squeeze(0)
    return tokenizer.detokenize(predicted_tokens.tolist())

if __name__ == "__main__":
    # Sample input for testing
    input_text = "hello world this is a test"
    response = get_response(input_text)
    print("Input:", input_text)
    print("Response:", response)

def train(model, dataset, epochs=10, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        tokens = self.tokenize(text)
        return torch.tensor(tokens)

    def tokenize(self, text):
        # Tokenization logic (you can use a tokenizer here)
        return [ord(c) for c in text]  # Example: convert chars to their ASCII values
