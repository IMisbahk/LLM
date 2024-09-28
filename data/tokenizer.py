class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}  # Reverse lookup

    def tokenize(self, text):
        return [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in text.split()]

    def idx_to_vocab(self, idx):
        return self.idx_to_word.get(idx, '<unk>')  # Return '<unk>' if index is not found
