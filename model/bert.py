import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.utils import PositionalEncoding, generate_square_subsequent_mask
from typing import Optional


class BERT(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
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
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, generate_square_subsequent_mask(src.size(-2))
        )
        # output = self.decoder(output)
        return output


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    layer = BERT(16, 2, 256, 2)
    src = torch.randn(256, 4, 16)
    print(src.device, layer(src).device)
