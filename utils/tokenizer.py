import torch

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(vocab)}

    def tokenize(self, text):
        return torch.tensor([self.word_to_index.get(word, self.word_to_index['<unk>']) for word in text.split()])

    def detokenize(self, indices):
        return ' '.join([self.index_to_word[idx] for idx in indices if idx in self.index_to_word])
