import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stt import STT
from model.gtrxl import GTrXL
from model.bert import BERT


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        act_embed_size,
        history_len,
        transformer_core: str = None,
    ):
        super(Critic, self).__init__()

        # Embedding Action in higher dimensional space
        self.action_embedded = nn.Linear(
            action_dim, act_embed_size
        )  # Come back to this

        self.transformer_core = transformer_core

        if self.transformer_core is None:
            self.l1 = nn.Linear(state_dim + act_embed_size, 256)
            self.l4 = nn.Linear(state_dim + act_embed_size, 256)
        else:
            valid_core = ["BERT", "GTrXL", "STT"]
            if self.transformer_core not in valid_core:
                raise ValueError("Invalid transformer core")

            # Continuous embedding
            self.embedding = nn.Linear(state_dim + act_embed_size, 256)
            self.l1 = nn.Linear(256, 256)
            self.l4 = nn.Linear(256, 256)

            # import transformer
            self.transformer = globals()[self.transformer_core](
                d_model=256,
                history_len=history_len,
                nhead=2,
                d_hid=256,
                nlayers=2,
                dropout=0.5,
                gate_type="residue",
                norm_first=False,
            )

            self.flatten = nn.Linear(256 * history_len, 256)

        # Q1 architecture
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        a = self.action_embedded(action)

        sa = torch.cat([state, a], -1)

        x = sa
        if self.transformer_core is not None:
            # Continuous embedding
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.flatten(torch.flatten(x, 1))
        else:
            x = x[:, 0, :]

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    history_len = 4
    a = Critic(16, 1, 16, history_len, transformer_core="STT")
    obs = torch.randn(5, history_len, 16)
    act = torch.randn(5, history_len, 1)
    q1, q2 = a(obs, act)
    print(q1.shape, q2.device)
