import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=32, num_heads=2, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, src, tgt):
        src = self.input_proj(src)
        tgt = self.input_proj(tgt)
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        return self.output_proj(out)