import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_layers=6, ff_hidden_dim=2048):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_heads, ff_hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) * (x.size(1) ** 0.5)  # Embedding scaling
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output)
        return output
