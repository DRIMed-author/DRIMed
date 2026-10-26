import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN


class Model(nn.Module):
    """MoleRec (NeurIPS 2022).

    Model combination/substructure effects of drugs with a multi-relational medication
    graph (EHR co-occurrence + DDI interaction), encode a patient state from diagnoses
    and procedures, and score each drug as the interaction between the patient state and
    the drug node embeddings.

    SMILES/molecular graphs are unavailable here, so the molecular-substructure modelling
    is approximated by GCNs over the EHR/DDI medication graphs.

    visit = [diag_codes, proc_codes, med_codes, age, gender]
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_med = vocab_size[2]

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # multi-relational medication graph encoders
        self.ehr_gcn = GCN(voc_size=self.num_med, emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=self.num_med, emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1).fill_(0.5))

        # diag + proc share one combined embedding table (as in the reference models)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.dropout = nn.Dropout(p=0.2)

        # temporal patient state encoder
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=False)

    def _visit_features(self, input):
        feat_seq = []
        for adm in input:
            entity_codes = list(adm[0]) + list(adm[1])
            if len(entity_codes) == 0:
                visit_feat = torch.zeros((1, self.emb_dim), device=self.device)
            else:
                entity_tensor = torch.LongTensor(entity_codes).unsqueeze(0).to(self.device)
                visit_feat = self.embedding(entity_tensor).mean(dim=1)
            feat_seq.append(self.dropout(visit_feat))
        return feat_seq

    def forward(self, input):
        feat_seq = self._visit_features(input)
        seq = torch.cat(feat_seq, dim=0).unsqueeze(1)  # (T, 1, dim)
        outputs, _ = self.gru(seq)  # (T, 1, dim)
        q = outputs[-1]  # (1, dim) current patient state

        # drug combination/substructure node embeddings
        med_node = self.ehr_gcn() + self.ddi_gcn() * self.inter  # (num_med, dim)

        result = torch.mm(q, med_node.t())  # (1, num_med)

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig  # (num_med, num_med)
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss