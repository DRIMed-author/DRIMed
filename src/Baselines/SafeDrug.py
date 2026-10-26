import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN


class Model(nn.Module):
    """SafeDrug (Yang et al. IJCAI 2021): dual-branch DDI-aware medication recommendation.

    A patient branch encodes the current visit's diag+proc entities into a patient
    representation, while the drug molecular branch (originally a SMILES + MPNN) is
    approximated by a GCN over the DDI graph, whose node embeddings serve as
    molecular-structure proxies. Per-drug logits are the inner product between the
    patient representation and the drug embeddings, complemented by the standard DDI
    regularization loss to penalize interacting drug co-predictions.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(p=0.2)

        # Patient branch: shared diag+proc embedding
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.type_embedding = nn.Embedding(2, emb_dim)
        self.type_gate = nn.Parameter(torch.tensor(0.02))

        # Drug molecular branch approximated by a GCN over the DDI graph
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.patient_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def _patient_representation(self, input):
        adm = input[-1]
        diag_codes = list(adm[0])
        proc_codes = list(adm[1])
        entity_codes = diag_codes + proc_codes
        if len(entity_codes) == 0:
            return torch.zeros((1, self.emb_dim), device=self.device)
        entity_tensor = torch.LongTensor(entity_codes).unsqueeze(0).to(self.device)
        type_ids = [0] * len(diag_codes) + [1] * len(proc_codes)
        type_tensor = torch.LongTensor(type_ids).unsqueeze(0).to(self.device)
        entity_emb = self.embedding(entity_tensor)
        entity_emb = entity_emb + torch.tanh(self.type_gate) * self.type_embedding(type_tensor)
        entity_emb = self.dropout(entity_emb)
        return entity_emb.mean(dim=1)

    def forward(self, input):
        patient_emb = self._patient_representation(input)  # (1, emb_dim)
        med_node = self.ddi_gcn()  # (num_med, emb_dim)

        patient_emb = self.patient_mlp(patient_emb)
        result = torch.mm(patient_emb, med_node.t())  # (1, num_med)

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss