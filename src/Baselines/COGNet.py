import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """COGNet (Copy-or-Generate, NeurIPS 2021).

    Predict the current medication set by combining two sources: (1) a copy module that
    attends over historical visits and re-selects their medications, and (2) a generate
    module that maps the current hidden state to novel medications. A learned pointer
    p_gen gates the two terms in logit space.

    visit = [diag_codes, proc_codes, med_codes, age, gender]
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_med = vocab_size[2]

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # diag + proc share one combined embedding table (as in the reference models)
        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        self.dropout = nn.Dropout(p=0.2)

        # temporal visit encoder
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=False)

        # generate module
        self.generate = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, self.num_med),
        )

        # copy-or-generate gate
        self.p_gen_linear = nn.Linear(emb_dim, 1)

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
        h_t = outputs[-1]  # (1, dim) current step hidden state

        # Generate
        generate_logits = self.generate(h_t)  # (1, num_med)
        p_gen = torch.sigmoid(self.p_gen_linear(h_t))  # (1, 1)

        # Copy
        if len(input) > 1:
            history_keys = outputs[:-1].squeeze(1)  # (T-1, dim)
            history_values = torch.zeros((len(input) - 1, self.num_med), device=self.device)
            for idx, adm in enumerate(input[:-1]):
                history_values[idx, adm[2]] = 1.0

            d_k = h_t.size(-1)
            attn_scores = torch.mm(h_t, history_keys.t()) / (d_k ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (1, T-1)
            copy_dist = torch.mm(attn_weights, history_values)  # (1, num_med)
        else:
            copy_dist = torch.zeros((1, self.num_med), device=self.device)

        # Fuse in logit space (numerically stable)
        copy_logits = torch.log(copy_dist + 1e-8)
        result = p_gen * generate_logits + (1.0 - p_gen) * copy_logits

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig  # (num_med, num_med)
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss