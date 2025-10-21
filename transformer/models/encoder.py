import torch
from torch import nn

from blocks.encoder_layer import EncoderLayer
# Impor komponen yang benar untuk masalah sinyal
from embedding.patch_embedding import PatchEmbedding
from embedding.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, num_patches, patch_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        
        # A. Logika Embedding sekarang ada di sini
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=num_patches + 1) # +1 untuk cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # B. Perbaiki bug rekursif: Panggil EncoderLayer, bukan Encoder
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                 ffn_hidden=ffn_hidden,
                                                 n_head=n_head,
                                                 drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, src, src_mask=None): # src_mask opsional, seringkali tidak perlu untuk klasifikasi
        # src shape: (batch_size, num_patches, patch_size)
        
        # 1. Terapkan patch embedding
        embedded_patches = self.patch_embedding(src) # -> (batch_size, num_patches, d_model)
        
        # 2. Tambahkan Class Token
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, embedded_patches], dim=1) # -> (batch_size, num_patches + 1, d_model)
        
        # 3. Tambahkan Positional Encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 4. Loop melalui setiap EncoderLayer
        for layer in self.layers:
            x = layer(x, src_mask) # src_mask harus disesuaikan jika digunakan

        return x
