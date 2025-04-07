import torch
import torch.nn as nn
import math

# class Embedding(nn.Module):
    
#     def __init__(self, d_model: int, size:int):
#         super().__init__()
#         self.d_model = d_model
#         self.size = size
#         self.embedding = nn.Embedding(size, d_model)
        
#     def forward(self,x):
#         return self.embedding(x) * math.sqrt(self.d_model)
    
# class PositionalEncoding(nn.Module):
     
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
            x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_model)
            
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
    
    def __init__(self, layers: nn.ModuleList)-> None:
        super().__init__()
        self.layers = layers
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
    
    def __init__(self, layers: nn.ModuleList)-> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

#-------
