import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """LEAP (Learning to Prescribe, Zhang et al. 2017): content + label attention over medication embeddings.

    A single-layer LSTM encodes the time-ordered visit representations (mean-pooled diag+proc embeddings)
    into a query h_T. Content attention softmax(h_T.E_m^T) captures diagnosis/procedure-to-medication
    relevance, while label attention modulates medication selection via historical medications' EHR
    co-occurrence (ehr_adj as the co-occurrence proxy). The combined weighted medication representation is
    dotted with E_m to produce per-medication logits. Simplified to work on the variable-length visit prefix.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(p=0.2)

        # combined diag + proc embedding (shared index space: diag codes + offset proc codes)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        # medication embedding matrix E_m
        self.med_embedding = nn.Embedding(vocab_size[2], emb_dim)

        # single-layer LSTM over visit embeddings
        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True)

        self.tensor_ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.label_scale = nn.Parameter(torch.tensor(0.5))

    def _visit_features(self, input):
        seq = []
        for adm in input:
            diag_codes = list(adm[0])
            proc_codes = list(adm[1])
            entity_codes = diag_codes + proc_codes
            if len(entity_codes) == 0:
                visit_feat = torch.zeros((1, self.emb_dim), device=self.device)
            else:
                entity_tensor = torch.LongTensor(entity_codes).unsqueeze(0).to(self.device)
                entity_emb = self.dropout(self.embedding(entity_tensor))
                visit_feat = entity_emb.mean(dim=1)
            seq.append(visit_feat)
        return seq

    def forward(self, input):
        visit_seq = self._visit_features(input)
        feat = torch.cat(visit_seq, dim=0).unsqueeze(0)  # (1, T, emb_dim)

        lstm_out, _ = self.lstm(feat)
        h = lstm_out[:, -1, :]  # (1, emb_dim)

        E_m = self.med_embedding.weight  # (num_med, emb_dim)
        d = self.emb_dim ** 0.5

        # content attention: relevance of each medication to the current clinical context
        content_w = F.softmax(torch.mm(h, E_m.t()) / d, dim=-1)  # (1, num_med)

        # label attention: co-occurrence with historical medications
        hist = torch.zeros((1, self.vocab_size[2]), device=self.device)
        for adm in input[:-1]:
            hist[0, list(adm[2])] = 1.0
        label_scores = torch.mm(hist, self.tensor_ehr_adj)  # (1, num_med)
        label_w = F.softmax(label_scores / d, dim=-1)

        combined_w = F.softmax(content_w + self.label_scale * label_w, dim=-1)
        fact = torch.mm(combined_w, E_m)  # (1, emb_dim)
        result = torch.mm(fact, E_m.t())  # (1, num_med)

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss