import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.utils import PositionalEncoding, generate_square_subsequent_mask
from typing import Optional


class GTrXLLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        n_layers=1,
        dim_feedforward=256,
        activation="relu",
        dropout=0,
        layer_norm_eps=1e-5,
        batch_first=False,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
        )
        self.gru1 = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.gru2 = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = (src).sum(dim=1).unsqueeze(dim=0)
        src = self.norm1(src)
        attn = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        emb, h = self.gru1(attn, h)

        mlp = self.norm2(emb)
        mlp = self.activation(self.linear1(mlp))
        mlp = self.activation(self.linear2(mlp))

        output, h = self.gru2(mlp, h)

        return output


class GTrXL(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        layer_norm_eps: float = 1e-5,
    ):
        super(GTrXL, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = GTrXLLayer(d_model, nhead, 1, dim_feedforward=d_hid)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers, encoder_norm
        )

        self.d_model = d_model

        # self.decoder = nn.Linear(d_model, ntoken)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, generate_square_subsequent_mask(src.size(0))
        )
        return output


if __name__ == "__main__":
    GTrXL(128, 128, 4, 128, 2)
