# main.py
import torch
import torch.nn as nn
from models.simple_language_model import SimpleLanguageModel
from utils.tokenizer import SimpleTokenizer

# Load or define vocabulary
vocab = ['<pad>', '<unk>', 'hello', 'world', 'this', 'is', 'a', 'test']
vocab_size = len(vocab)
embedding_dim = 100  # Adjust as necessary
hidden_dim = 128  # Adjust as necessary

# Initialize tokenizer and model
tokenizer = SimpleTokenizer(vocab)
model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Function to train the model (use later in the UI)
def train_model(model, inputs, targets, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

# Placeholder for training or other functionalities if needed
# Keep this file modular; actual chatbot functionality will be handled in chatbot_ui.py
