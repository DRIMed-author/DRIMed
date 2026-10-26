import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """RETAIN (Reverse Time Attention, Choi et al. 2016): two-level attention with bidirectional temporal modeling.

    Each visit's diag+proc codes are embedded and combined with a code-level attention beta_ij. Forward and
    backward GRUs process the visit-vector sequence to build visit-level hidden states, from which a
    visit-level attention alpha_i is derived. The attended context is fed to an MLP for medication logits.
    Simplified to variable-length visit prefixes and shared diag/proc embedding space.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(p=0.2)

        # combined diag + proc embedding (shared index space: diag codes + offset proc codes)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)

        # code-level attention context vector
        self.code_ctx = nn.Parameter(torch.randn(emb_dim))

        # bidirectional temporal modeling via forward (early->late) and backward (late->early) GRUs
        self.gru_f = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.gru_b = nn.GRU(emb_dim, emb_dim, batch_first=True)

        # visit-level attention over concatenated forward/backward hidden states
        self.attn_visit = nn.Linear(emb_dim * 2, 1)

        # prediction head
        self.output_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

    def _code_embeddings(self, input):
        code_vecs = []
        for adm in input:
            entity_codes = list(adm[0]) + list(adm[1])
            if len(entity_codes) == 0:
                code_vecs.append(torch.zeros((1, self.emb_dim), device=self.device))
            else:
                entity_tensor = torch.LongTensor(entity_codes).to(self.device)
                code_vecs.append(self.dropout(self.embedding(entity_tensor)))  # (K, emb_dim)
        return code_vecs

    def forward(self, input):
        code_vecs = self._code_embeddings(input)

        # code-level attention beta_ij within each visit
        visit_vecs = []
        for vec in code_vecs:
            scores = torch.mv(vec, self.code_ctx)  # (K,)
            beta = F.softmax(scores, dim=0)  # (K,)
            v_i = (beta.unsqueeze(1) * vec).sum(dim=0, keepdim=True)  # (1, emb_dim)
            visit_vecs.append(v_i)
        v = torch.cat(visit_vecs, dim=0).unsqueeze(0)  # (1, T, emb_dim)

        # bidirectional temporal encoding
        h_f, _ = self.gru_f(v)
        h_b, _ = self.gru_b(torch.flip(v, dims=[1]))
        h_b = torch.flip(h_b, dims=[1])
        h = torch.cat([h_f, h_b], dim=-1)  # (1, T, 2*emb_dim)

        # visit-level attention alpha_i
        alpha_scores = self.attn_visit(h).squeeze(-1)  # (1, T)
        alpha = F.softmax(alpha_scores, dim=-1)  # (1, T)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (1, 2*emb_dim)

        result = self.output_head(context)  # (1, num_med)

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss