import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN


class Model(nn.Module):
    """CausalMed: causal medication recommendation via backdoor adjustment.

    Diagnosis is treated as the confounder that affects both the clinical state and
    the prescription. The confounder sub-network projects diagnosis embeddings into an
    adjustment term g, and the de-confounded query is computed as q = h - sigmoid(gate)*g,
    where h is the temporal patient state (GRU over visit features). q then attends over a
    medication memory bank (EHR/DDI GCN fusion) to obtain (1, num_med) logits.

    Note: the original CausalMed uses intervention/do-calculus on a causal graph; here we
    approximate it with the gated residual de-confounding term, and use GCN medication
    memory to substitute any graph-encoded drug features.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(p=0.2)

        # medication memory bank (EHR / DDI graphs)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1).fill_(0.1))
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # combined diag/proc embedding (diag and proc share one index space)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.type_embedding = nn.Embedding(2, emb_dim)
        self.type_gate = nn.Parameter(torch.tensor(0.02))

        # temporal patient state
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        # diagnosis-as-confounder sub-network
        self.confounder = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Tanh())
        self.deconf_gate = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid())

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
        h = gru_out[:, -1, :]  # (1, dim), current patient state

        # diagnosis-as-confounder adjustment term g
        diag_codes = list(input[-1][0])
        if len(diag_codes) == 0:
            g = torch.zeros((1, self.emb_dim), device=self.device)
        else:
            diag_tensor = torch.LongTensor(diag_codes).unsqueeze(0).to(self.device)
            g = self.confounder(self.embedding(diag_tensor)).mean(dim=1)

        # de-confounded query: q = h - sigmoid(gate) * g
        gate = self.deconf_gate(torch.cat([h, g], dim=-1))
        q = h - gate * g

        drug_memory = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.inter)

        key_weights = F.softmax(torch.mm(q, drug_memory.t()) / 0.5, dim=-1)
        fact_final = torch.mm(key_weights, drug_memory)

        result = self.output_head(torch.cat([q, fact_final], dim=-1))

        sig_result = torch.sigmoid(result)
        neg_pred_prob = sig_result.t() * sig_result
        batch_neg = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg