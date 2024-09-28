import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_length, d_model)
        x = x.permute(1, 0, 2)  # Transformer expects shape (seq_length, batch_size, d_model)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # Shape: (batch_size, seq_length, d_model)
        return self.fc_out(x)
