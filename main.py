import json
import torch
from models.transformer_model import TransformerModel
from utils.tokenizer import SimpleTokenizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re

def build_vocab(file_path):
    vocab = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = re.findall(r'\b\w+\b', line.lower())
            vocab.update(words)
    vocab = ['<pad>', '<unk>'] + list(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()
        self.tokenizer = SimpleTokenizer(build_vocab(file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        tokens = self.tokenize(text)
        return torch.tensor(tokens)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

def train(model, dataset, epochs=10, batch_size=32):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

def get_response(message, tokenizer, model):
    # Tokenize the input message
    input_tokens = tokenizer.tokenize(message)  # Tokenize using your SimpleTokenizer
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)  # Convert to tensor and add batch dimension
    
    output = model(input_tensor)

    # Get the predicted tokens
    predicted_tokens = output.argmax(dim=-1).squeeze(0)
    return tokenizer.detokenize(predicted_tokens.tolist())


if __name__ == "__main__":
    dataset = TextDataset('data/dataset.txt')  # Provide your actual file path here
    vocab = dataset.tokenizer.vocab
    vocab_size = len(vocab)

    model = TransformerModel(vocab_size)  # Ensure your TransformerModel is defined correctly
    train(model, dataset, epochs=10, batch_size=32)

    # Sample input for testing
    input_text = "hello world this is a test"
    response = get_response(input_text, dataset.tokenizer, model)
    print("Input:", input_text)
    print("Response:", response)
