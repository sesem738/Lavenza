import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.utils import (
    PositionalEncoding,
    ResidueGate,
    GRUGate,
    generate_square_subsequent_mask,
)
from torch import Tensor
from typing import Optional


class STTLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        history_len,
        nhead,
        dim_feedforward=256,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
        gate_type="residue",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            norm_first,
            layer_norm_eps,
        )
        self.time_sa_block = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.space_sa_block = nn.MultiheadAttention(
            history_len, nhead, dropout=dropout, batch_first=batch_first
        )
        self.norm_time = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_space = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_space_first = nn.LayerNorm(history_len, eps=layer_norm_eps)

        gate_dict = {"residue": ResidueGate, "gru": GRUGate}
        self.gate = gate_dict[gate_type]()
        self.resgate = ResidueGate()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.MultiheadAttention)):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.in_proj_bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x_time = self.gate(
                x,
                self._sa_block(
                    self.time_sa_block,
                    self.norm_time(x),
                    src_mask,
                    src_key_padding_mask,
                ),
            )
            x_space = self.gate(
                x,
                self._sa_block(
                    self.space_sa_block,
                    self.norm_space_first(x.permute(0, 2, 1)),
                    None,
                    None,
                ).permute(0, 2, 1),
            )  # No masking for spatial attention
            x = x_time + x_space
            x = self.resgate(x, self._ff_block(self.norm2(x)))
        else:
            x_time = self.norm_time(
                self.gate(
                    x,
                    self._sa_block(
                        self.time_sa_block, x, src_mask, src_key_padding_mask
                    ),
                )
            )
            x_space = self.norm_space(
                self.gate(
                    x,
                    self._sa_block(
                        self.space_sa_block, x.permute(0, 2, 1), None, None
                    ).permute(0, 2, 1),
                )
            )  # No masking for spatial attention
            x = x_time + x_space
            # print(self.activation, self.linear1, self.linear2)
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        sa_block: nn.Module,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = sa_block(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)


class STT(nn.Module):
    def __init__(
        self,
        d_model: int,
        history_len: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        layer_norm_eps: float = 1e-5,
        gate_type="residue",
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = STTLayer(
            d_model,
            history_len,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            gate_type=gate_type,
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

    layer = STT(16, 4, 2, 256, 2, gate_type="residue")
    src = torch.randn(256, 4, 16)
    print(src.device, layer(src).device)
