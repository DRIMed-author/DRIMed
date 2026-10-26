import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN


class Model(nn.Module):
    """MR-DTR: multi-relational drug recommendation via relation-aware drug graphs.

    Two relations are modeled on the medication graph: drug-drug interaction (ddi_adj) and
    drug co-occurrence (ehr_adj). Each relation is encoded by an independent GCN, and the two
    node embeddings are fused into a single medication node embedding med_node (num_med x dim).
    The patient visit sequence (mean-pooled diag/proc combined embeddings) is encoded by a GRU;
    the current hidden state q attends over med_node to output (1, num_med) logits.

    Note: the original MR-DTR uses multi-relational graph convolution (R-GCN). Here we
    approximate it with independent GCNs per relation followed by a learned fusion layer.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(p=0.2)

        # multi-relational medication graph: DDI relation + co-occurrence relation
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.rel_fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
        )
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # combined diag/proc embedding (diag and proc share one index space)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.type_embedding = nn.Embedding(2, emb_dim)
        self.type_gate = nn.Parameter(torch.tensor(0.02))

        # temporal patient sequence
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        # medication prediction head
        self.output_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
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
        feat = torch.cat(visit_feat_seq, dim=0).unsqueeze(0)  # (1, T, dim)

        gru_out, _ = self.gru(feat)
        q = gru_out[:, -1, :]  # (1, dim), current patient state

        # fuse the two relation-specific node embeddings
        med_node = self.rel_fuse(torch.cat([self.ddi_gcn(), self.ehr_gcn()], dim=-1))  # (num_med, dim)

        key_weights = F.softmax(torch.mm(q, med_node.t()) / 0.5, dim=-1)
        fact_final = torch.mm(key_weights, med_node)

        result = self.output_head(torch.cat([q, fact_final], dim=-1))

        sig_result = torch.sigmoid(result)
        neg_pred_prob = sig_result.t() * sig_result
        batch_neg = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg