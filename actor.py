from xmlrpc.client import Boolean
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stt import STT
from model.gtrxl import GTrXL
from model.bert import BERT

"""
1) Project observation into embedding space
2) Add Position encoding
3) 

"""


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        history_len,
        transformer_core: str = None,
    ):
        super(Actor, self).__init__()

        self.transformer_core = transformer_core

        if self.transformer_core is None:
            self.l1 = nn.Linear(state_dim, 256)
        else:
            valid_core = ["BERT", "GTrXL", "STT"]
            if self.transformer_core not in valid_core:
                raise ValueError("Invalid transformer core")

            # Continuous embedding
            self.embedding = nn.Linear(state_dim, 256)
            self.l1 = nn.Linear(256, 256)

            # import transformer
            self.transformer = globals()[self.transformer_core](
                d_model=256,
                history_len=history_len,
                nhead=1,
                d_hid=256,
                nlayers=2,
                dropout=0.5,
                gate_type="residue",
                norm_first=False,
            )

            self.flatten = nn.Linear(256 * history_len, 256)

        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = state
        if self.transformer_core is not None:
            # Continuous embedding
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.flatten(torch.flatten(x, 1))
        else:
            x = x[:, -1, :]

        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    history_len = 4
    a = Actor(16, 1, 1, history_len, transformer_core="STT")
    src = torch.randn(5, history_len, 16)
    b = a(src)
    print(b.shape, b.device)
