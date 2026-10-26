import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """RaVSNet baseline: Reconstruct & View Sequence Network for medication recommendation.

    Core idea: each visit is represented by mean-pooling its diag+proc code embeddings; the
    current visit is *reconstructed* from the history through an attention over past visits,
    and the visit sequence is encoded with a GRU into a *view* representation. The reconstruction
    and view are fused and passed through an MLP to predict the medication logits. As we only
    have diagnosis/procedure entities, both the reconstruction keys/values and the view input
    share the same pooled visit vectors; for a single visit the reconstruction degenerates to the
    current visit itself.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # Diag + proc share one embedding table (proc codes are already offset by the loader).
        self.entity_emb = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.dropout = nn.Dropout(p=0.2)

        # View sequence module: GRU over the visit sequence.
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        # Fusion of reconstructed representation r and view representation v.
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

    def _visit_features(self, input):
        seq = []
        for adm in input:
            codes = list(adm[0]) + list(adm[1])
            if len(codes) == 0:
                vec = torch.zeros(1, self.emb_dim, device=self.device)
            else:
                idx = torch.LongTensor(codes).to(self.device)
                vec = self.entity_emb(idx).mean(dim=0, keepdim=True)
                vec = self.dropout(vec)
            seq.append(vec)
        return torch.cat(seq, dim=0)  # (num_visits, emb_dim)

    def forward(self, input):
        visits = self._visit_features(input)  # (T, emb_dim)

        # Reconstruct module: current visit attended over historical visits.
        cur = visits[-1:]  # (1, emb_dim)
        if visits.size(0) > 1:
            history = visits[:-1]  # (T-1, emb_dim)
            scores = torch.mm(cur, history.t()) / (self.emb_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)
            r = torch.mm(weights, history)  # (1, emb_dim)
        else:
            r = cur

        # View sequence module: GRU encodes the visit sequence, take the current hidden view.
        out, _ = self.gru(visits.unsqueeze(0))  # (1, T, emb_dim)
        v = out[:, -1, :]  # (1, emb_dim)

        result = self.fusion(torch.cat([r, v], dim=-1))  # (1, num_med)

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss