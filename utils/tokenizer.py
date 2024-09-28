class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_index = {word: i for i, word in enumerate(vocab)}

    def tokenize(self, text):
        return [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in text.lower().split()]

    def detokenize(self, tokens):
        return ' '.join([self.vocab[token] for token in tokens if token < len(self.vocab)])
