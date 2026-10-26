import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN
from src.utils.encoder import DRIMedEncoder


class Model(nn.Module):
    """DRIMed w/o HMR: remove the historical medication retrieval module (no history attention / context gate)."""

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.dropout = nn.Dropout(p=0.2)

        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1).fill_(0.1))
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)
        self.type_embedding = nn.Embedding(2, self.emb_dim_ff)
        self.type_gate = nn.Parameter(torch.tensor(0.02))

        self.samformer = DRIMedEncoder(num_channels=1, seq_len=self.emb_dim_ff, hid_dim=self.emb_dim, use_revin=False)

        self.cnn1d = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.emb_dim, kernel_size=3, padding=1, groups=self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.cnn1d_k5 = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.emb_dim, kernel_size=5, padding=2, groups=self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.cnn_scale_gate = nn.Parameter(torch.tensor(0.05))
        self.cnn_ln = nn.LayerNorm(self.emb_dim)
        self.ln_feat = nn.LayerNorm(self.emb_dim)

        self.age_proj = nn.Linear(1, emb_dim)
        self.gender_emb = nn.Embedding(2, emb_dim)
        self.demo_gate = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid())
        self.demo_correction = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.Tanh())

        self.output_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

    def _visit_features(self, input):
        visit_feat_seq, demo_list = [], []
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
                entity_feat = self.samformer(entity_emb)
                visit_feat = entity_feat.mean(dim=1)
            visit_feat_seq.append(visit_feat)

            age = torch.tensor([[adm[3]]], dtype=torch.float, device=self.device)
            gender = torch.tensor([adm[4]], dtype=torch.long, device=self.device)
            demo_list.append(torch.cat([self.age_proj(age), self.gender_emb(gender)], dim=-1))
        return visit_feat_seq, demo_list

    def forward(self, input):
        visit_feat_seq, demo_list = self._visit_features(input)

        feat = torch.cat(visit_feat_seq, dim=0).unsqueeze(0)

        feat_cnn_input = feat.transpose(1, 2)
        feat = self.cnn1d(feat_cnn_input) + torch.tanh(self.cnn_scale_gate) * self.cnn1d_k5(feat_cnn_input)
        feat = feat.transpose(1, 2)
        feat = self.cnn_ln(feat).squeeze(0)
        feat = self.ln_feat(feat)

        query = feat[-1:]
        demo_emb = demo_list[-1]

        gate = self.demo_gate(demo_emb)
        correction = self.demo_correction(demo_emb)
        query = query + gate * correction

        drug_memory = self.ehr_gcn() - self.ddi_gcn() * torch.sigmoid(self.inter)

        # no historical medication retrieval: query retrieves memory directly
        key_weights = F.softmax(torch.mm(query, drug_memory.t()) / 0.5, dim=-1)
        fact_final = torch.mm(key_weights, drug_memory)

        result = self.output_head(torch.cat([query, fact_final], dim=-1))

        sig_result = torch.sigmoid(result)
        neg_pred_prob = sig_result.t() * sig_result
        batch_neg = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg