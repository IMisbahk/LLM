# utils/tokenizer.py
import torch
# utils/tokenizer.py
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_token = {idx: word for idx, word in enumerate(vocab)}

    def tokenize(self, text):
        """Tokenizes the input text into a tensor of token IDs."""
        tokens = text.split()  # Simple whitespace tokenizer
        token_ids = [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
        return torch.tensor(token_ids)  # Convert to tensor

    def detokenize(self, token_ids):
        """Converts token IDs back to the original text."""
        tokens = [self.id_to_token.get(idx, '<unk>') for idx in token_ids]
        return ' '.join(tokens)
