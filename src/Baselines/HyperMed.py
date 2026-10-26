import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.layers import GCN

CURV = 1.0  # Poincare ball curvature, radius = 1 / sqrt(CURV)
BALL_EPS = 1e-5


def _project(x, c=CURV):
    """Clamp points into the open Poincare ball (||x|| < 1 / sqrt(c) - eps)."""
    radius = 1.0 / math.sqrt(c) - BALL_EPS
    norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
    return torch.where(norm > radius, x / norm * radius, x)


def _mobius_add(x, y, c=CURV):
    """Mobius addition on the Poincare ball."""
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return _project(num / den.clamp_min(1e-12), c)


def _conformal(x, c=CURV):
    x2 = (x * x).sum(-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp_min(BALL_EPS)


def _exp_map(x, v, c=CURV):
    """Exponential map at point x of tangent vector v."""
    v_norm = v.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
    lam = _conformal(x, c)
    arg = torch.tanh(math.sqrt(c) * lam * v_norm / 2.0) * (v / (math.sqrt(c) * v_norm))
    return _mobius_add(x, arg, c)


def _log_map(x, y, c=CURV):
    """Logarithmic map of y at point x (maps back to the tangent space at x)."""
    diff = _mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
    lam = _conformal(x, c)
    bound = 1.0 / math.sqrt(c) - BALL_EPS
    diff_norm = diff_norm.clamp(max=bound - 1e-9)
    fac = (2.0 / (math.sqrt(c) * lam)) * torch.atanh(math.sqrt(c) * diff_norm)
    return fac * diff


def _einstein_midpoint(xs, weights=None, c=CURV):
    """Einstein midpoint (Lorentz-model centroid) of points in the Poincare ball."""
    xs = _project(xs, c)
    n = xs.size(0)
    if weights is None:
        weights = xs.new_full((n,), 1.0 / n)
    x2 = (xs * xs).sum(-1)
    gamma = 1.0 / torch.sqrt((1.0 - c * x2).clamp_min(BALL_EPS))
    w = gamma * weights
    num = (w.unsqueeze(-1) * xs).sum(0, keepdim=True)
    return _project(num / w.sum().clamp_min(1e-12), c)


class Model(nn.Module):
    """HypeMed baseline: hyperbolic (Poincare ball) medication recommendation.

    Core idea: embed codes and drugs in the Poincare ball to capture their hierarchy, aggregate a
    visit's diag+proc embeddings with the Einstein midpoint, then aggregate the visit set into a
    patient representation (hierarchical aggregation). The patient point is mapped back to the
    tangent space via the log map and decoded through an MLP whose final weight matrix is the
    tangent-space drug embeddings. Since the repo provides no SMILES/molecular graphs, the drug
    embeddings come from a drug-graph GCN over ehr_adj (approximating molecule graphs) and are
    pushed into the ball via the exponential map. All points are clamped to ||x|| < 1 - eps.
    """

    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0')):
        super(Model, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.c = CURV

        # Diag + proc share one embedding table (proc codes are already offset by the loader).
        self.entity_emb = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, emb_dim)
        nn.init.normal_(self.entity_emb.weight, mean=0.0, std=0.02)

        # Drug embeddings: drug-graph GNN approximating the molecular graphs of the original paper.
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)

        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.final_bias = nn.Parameter(torch.zeros(vocab_size[2]))

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

    def forward(self, input):
        origin = torch.zeros(1, self.emb_dim, device=self.device)

        # Visit-level hyperbolic aggregation: Einstein midpoint of each visit's diag+proc codes.
        visit_hyp = []
        for adm in input:
            codes = list(adm[0]) + list(adm[1])
            if len(codes) == 0:
                visit_hyp.append(origin.clone())
            else:
                idx = torch.LongTensor(codes).to(self.device)
                ents = _project(self.entity_emb(idx), self.c)  # (n, emb_dim)
                visit_hyp.append(_einstein_midpoint(ents, None, self.c))  # (1, emb_dim)
        visits = torch.cat(visit_hyp, dim=0)  # (T, emb_dim)

        # Patient-level hyperbolic aggregation over the visit set.
        patient_hyp = _einstein_midpoint(visits, None, self.c)  # (1, emb_dim)
        patient_euclid = _log_map(origin, patient_hyp, self.c)  # (1, emb_dim), tangent space

        # Drug Poincare embeddings from the drug-graph GNN, mapped into/back from the ball.
        origin_drug = origin.expand(self.vocab_size[2], -1)
        drug_tangent = self.ehr_gcn()  # (num_med, emb_dim)
        drug_hyp = _exp_map(origin_drug, drug_tangent, self.c)
        drug_euclid = _log_map(origin_drug, drug_hyp, self.c)  # (num_med, emb_dim)

        hidden = F.relu(self.out_proj(self.dropout(patient_euclid)))
        result = torch.mm(hidden, drug_euclid.t()) + self.final_bias  # (1, num_med)

        sig = torch.sigmoid(result)
        neg_pred_prob = sig.t() * sig
        ddi_loss = 0.0003 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, ddi_loss