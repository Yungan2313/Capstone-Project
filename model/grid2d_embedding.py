import torch
import torch.nn as nn

class Grid2DEmbedding(nn.Module):
    def __init__(self,
                 num_cells_x: int,
                 num_cells_y: int,
                 embedding_dim: int,
                 dropout: float = 0.0,
                 use_layernorm: bool = False,
                 mode: str = "concat"):
        """
        Args:
            num_cells_x: 經度方向格子總數
            num_cells_y: 緯度方向格子總數
            embedding_dim: 每個軸的嵌入維度
            dropout: dropout 機率
            use_layernorm: 是否使用 LayerNorm
            mode: 'sum' or 'concat'
        """
        super(Grid2DEmbedding, self).__init__()
        assert mode in ["sum", "concat"], "mode 必須為 'sum' 或 'concat'"
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.use_layernorm = use_layernorm

        self.embedding_x = nn.Embedding(num_cells_x, embedding_dim)
        self.embedding_y = nn.Embedding(num_cells_y, embedding_dim)

        output_dim = embedding_dim if mode == "sum" else embedding_dim * 2
        self.dropout = nn.Dropout(dropout)

        if use_layernorm:
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, grid_x, grid_y):
        """
        Args:
            grid_x: Tensor [B, L]，經度格子 index
            grid_y: Tensor [B, L]，緯度格子 index
            B = batch size, L = sequence length
        Returns:
            Tensor [B, L, D] 或 [B, L, 2D]，視 mode 而定
        """
        x_emb = self.embedding_x(grid_x)  # [B, L, D]
        y_emb = self.embedding_y(grid_y)  # [B, L, D]

        if self.mode == "sum":
            emb = x_emb + y_emb  # [B, L, D]
        elif self.mode == "concat":
            emb = torch.cat([x_emb, y_emb], dim=-1)  # [B, L, 2D]

        # if self.use_layernorm:
        #     emb = self.norm(emb)

        # emb = self.dropout(emb)
        return emb

if __name__ == "__main__":
    # 如果要使用切記這邊沒有將x y轉換成grid_x grid_y，需要自己寫好
    batch_size = 32
    seq_len = 128

    # 測試 concat 模式
    embedder = Grid2DEmbedding(
        num_cells_x=11361,
        num_cells_y=7164,
        embedding_dim=64,
        dropout=0.1,
        use_layernorm=True,
        mode="concat"
    )

    grid_x = torch.randint(0, 11361, (batch_size, seq_len))
    grid_y = torch.randint(0, 7164, (batch_size, seq_len))

    out = embedder(grid_x, grid_y)
    print("輸出 shape:", out.shape)  # 應該是 [32, 128, 128] (64 x/y concat)

    # 如果你要餵進 transformer，外面再接 linear 映射到 model_dim 即可
