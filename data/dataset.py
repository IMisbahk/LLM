import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=50):
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx].strip()
        tokens = self.tokenize(text)

        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.vocab.index('<pad>')] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return torch.tensor(tokens)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
