import torch
from models.transformer_model import TransformerModel
from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset  
from training.train import train  

def load_vocab(file_path):
    vocab = ['<pad>', '<unk>'] 
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if word not in vocab:
                    vocab.append(word)
    return vocab

def get_response(message, tokenizer, model):
    input_tokens = tokenizer.tokenize(message)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)  

    with torch.no_grad():
        output = model(input_tensor)    

        if output.dim() == 3:
            probabilities = torch.softmax(output[:, -1, :], dim=-1) 
        elif output.dim() == 2:
            probabilities = torch.softmax(output, dim=-1)  
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

    if probabilities.dim() != 2:
        raise ValueError(f"Probabilities must be 1 or 2D but got {probabilities.dim()}D")

    predicted_token = torch.multinomial(probabilities, num_samples=1).item()
    return tokenizer.idx_to_vocab(predicted_token)

if __name__ == "__main__":
    vocab = load_vocab('data/dataset.txt') 
    vocab_size = len(vocab)
    tokenizer = SimpleTokenizer(vocab)
    model = TransformerModel(vocab_size)

    model.eval()

    input_text = "hello world"
    response = get_response(input_text, tokenizer, model)
    print("Input:", input_text)
    print("Response:", response)

    dataset = TextDataset('data/dataset.txt', tokenizer)  
    train(model, dataset, epochs=50, batch_size=32)
