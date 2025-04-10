import torch
import torch.nn as nn
import math

# class Embedding(nn.Module):
    
    # def __init__(self, d_model: int, size:int):
    #     super().__init__()
    #     self.d_model = d_model
    #     self.size = size
    #     self.embedding = None
        
    # def trajembedding(self, x):
    #     # x => shape(40,2)
        
        
    # def forward(self,x):
    #     return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int = 2000, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
     
class LayerNormalization(nn.Module):
    
    def __init__(self,eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return x
    
    
class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        #(Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
        
        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.dropout = nn.Dropout(dropout)
            
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            
            self.d_k = d_model // n_heads
            
            self.w_q = nn.Linear(d_model, d_model) #Wq make every paremeter a weight to be learned
            self.w_k = nn.Linear(d_model, d_model) #Wk
            self.w_v = nn.Linear(d_model, d_model) #Wv
            
            self.w_out = nn.Linear(d_model, d_model) #Wo
            self.dropout = nn.Dropout(dropout)
            
        @staticmethod
        def attention(query, key, value, mask, dropout: nn.Dropout):
            d_k = query.size(-1) #d_k = d_model / n_heads
            
            attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (Batch, n_heads, Seq, d_k) @ (Batch, n_heads, d_k, Seq) -> (Batch, n_heads, Seq, Seq)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores = attention_scores.softmax(dim=-1) # (Batch, n_heads, Seq, Seq)
            if dropout is not None:
                attention_scores = dropout(attention_scores)
                
            return (attention_scores @ value), attention_scores # (Batch, n_heads, Seq, d_k) @ (Batch, n_heads, Seq, d_k) -> (Batch, n_heads, Seq, d_model), attention_scores (Batch, n_heads, Seq, Seq)
            
        def forward(self, q, k, v, mask=None): #mask=None => Multi-Head Attention block, mask=True => Masked Multi-Head Attention block
            query = self.w_q(q) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
            key = self.w_k(k)
            value = self.w_v(v)
            #seq = length of the input, batch = batch size, d_model = embedding size, d_k = d_model / n_heads
            query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2) # (Batch, Seq, d_model) -> (Batch, n_heads, Seq, d_k) make it can use to different heads
            key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2) # (Batch, Seq, d_model) -> (Batch, n_heads, Seq, d_k)
            value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)
            
            x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) # (Batch, n_heads, Seq, d_k) -> (Batch, Seq, d_model)
            
            # (Batch, n_heads, Seq, d_model) -> (Batch, Seq, n_heads, d_model) -> (Batch, Seq, d_model)
            x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)
            
            return self.w_out(x) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    
    def __init__(self, self_attention_blocks: MultiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attention_blocks = self_attention_blocks
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_blocks(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    
    def __init__(self, self_attention_blocks: MultiHeadAttention, cross_attention_blocks: MultiHeadAttention, feed_forward: FeedForward, dropout: float = 0.1)-> None:
        super().__init__()
        self.self_attention_blocks = self_attention_blocks
        self.cross_attention_blocks = cross_attention_blocks
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_blocks(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_blocks(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class FinalMLP(nn.Module):
    
    def __init__(self, d_model: int, output_dim: int, dropout: float = 0.1)-> None:
        super().__init__()
        self.linear = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, output_dim)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, FinalMLP: FinalMLP, d_input=2, d_model = 32) -> None:
        super().__init__()
        self.Input_Add_dimension = nn.Linear(2,d_model)
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.FinalMLP = FinalMLP

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.Input_Add_dimension(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.Input_Add_dimension(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def FMLP(self, x):
        # (batch, seq_len, d_model)
        return self.FinalMLP(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        out = self.decode(memory, src_mask, tgt, tgt_mask)
        return self.FMLP(out)

def build_transformer(N=4, d_model=32, d_ff=128, h=4, dropout=0.1):
    self_attn = MultiHeadAttention(d_model, h, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)

    enc_layer = EncoderLayer(self_attn, feed_forward, dropout)
    dec_layer = DecoderLayer(self_attn, self_attn, feed_forward, dropout)

    encoder = Encoder(enc_layer, N)
    decoder = Decoder(dec_layer, N)

    src_pos = PositionalEncoding(d_model)
    tgt_pos = PositionalEncoding(d_model)
    output_proj = FinalMLP(d_model, output_dim=2)

    return Transformer(encoder, decoder, src_pos, tgt_pos, output_proj, d_input=2, d_model=d_model)