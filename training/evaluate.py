import torch
from torch.utils.data import DataLoader
from models.transformer_model import TransformerModel
from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset  
from main import load_vocab

vocab = load_vocab('data/dataset.txt') 
vocab_size = len(vocab)
tokenizer = SimpleTokenizer(vocab)

dataset = TextDataset('data/dataset.txt', tokenizer)
batch_size = 32 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = TransformerModel(vocab_size)
model.eval() 

def evaluate_model(model, dataloader):
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  
    for inputs in dataloader:
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average loss: {avg_loss}")

if __name__ == "__main__":
    evaluate_model(model, dataloader)
