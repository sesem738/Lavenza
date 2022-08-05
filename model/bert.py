import torch
import numpy as np
from  torch.nn import Module, TransformerEncoderLayer, TransformerEncoder, LayerNorm
import torch.nn.functional as F
from utils import PositionalEncoding, generate_square_subsequent_mask
from typing import Optional


class BERT(Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = TransformerEncoder(
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
        print(output.shape)
        # output = self.decoder(output)
        return output

if __name__ == "__main__":
    BERT(128,128,4,128,2)