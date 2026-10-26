import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN


class Model(nn.Module):
    """GAMENet (Graph Augmented MEmory Network, Shang et al. KDD 2019).

    A dual memory network for medication recommendation. The drug memory keys are
    GCN node embeddings built over the EHR/DDI medication graphs, the value memory
    stores the multi-hot historical medications, and the query is the mean-pooled
    diag+proc embedding of the current visit. Attention reads over the drug memory
    (fact1) and over historical visits weighted by their medications (fact2), and a
    standard DDI regularization penalizes co-prediction of interacting drugs.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(p=0.2)

        # Shared diag+proc embedding (combined index space)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.type_embedding = nn.Embedding(2, emb_dim)
        self.type_gate = nn.Parameter(torch.tensor(0.02))

        # Medication memory keys over EHR / DDI graphs
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1).fill_(0.1))
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # Value-memory projection for historical medications
        self.history_proj = nn.Linear(vocab_size[2], emb_dim)

        # Output head over [query, fact1, fact2]
        self.output_head = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

    def _visit_features(self, input):
        visit_feat_seq = []
        for adm in input:
            diag_codes = list(adm[0])
            proc_codes = list(adm[1])
            entity_codes = diag_codes + proc_codes
            if len(entity_codes) == 0:
                visit_feat = torch.zeros((1, self.emb_dim), device=self.device)
            else:
                entity_tensor = torch.LongTensor(entity_codes).unsqueeze(0).to(self.device)
                type_ids = [0] * len(diag_codes) + [1] * len(proc_codes)
                type_tensor = torch.LongTensor(type_ids).unsqueeze(0).to(self.device)
                entity_emb = self.embedding(entity_tensor)
                entity_emb = entity_emb + torch.tanh(self.type_gate) * self.type_embedding(type_tensor)
                entity_emb = self.dropout(entity_emb)
                visit_feat = entity_emb.mean(dim=1)
            visit_feat_seq.append(visit_feat)
        return visit_feat_seq

    def forward(self, input):
        visit_feat_seq = self._visit_features(input)
        query = visit_feat_seq[-1]  # (1, emb_dim)

        drug_memory = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.inter)  # (num_med, emb_dim)

        d_k = query.size(-1)
        key_weights = F.softmax(torch.mm(query, drug_memory.t()) / (d_k ** 0.5), dim=-1)
        fact1 = torch.mm(key_weights, drug_memory)  # (1, emb_dim)

        if len(input) > 1:
            history_keys = torch.cat(visit_feat_seq[:-1], dim=0)  # (num_visits-1, emb_dim)
            history_values = torch.zeros((len(input) - 1, self.vocab_size[2]), device=self.device)
            for idx, adm in enumerate(input[:-1]):
                history_values[idx, adm[2]] = 1.0

            attn_scores = torch.mm(query, history_keys.t()) / (d_k ** 0.5)
            visit_weight = F.softmax(attn_scores, dim=-1)
            weighted_values = torch.mm(visit_weight, history_values)  # (1, num_med)
            fact2 = F.relu(self.history_proj(weighted_values))  # (1, emb_dim)
        else:
            fact2 = fact1  # degenerate to the drug memory read for single visits

        result = self.output_head(torch.cat([query, fact1, fact2], dim=-1))

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss