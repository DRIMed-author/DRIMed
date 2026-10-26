import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization (https://openreview.net/pdf?id=cGDAkQo1C0p)."""

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class DRIMedEncoder(nn.Module):
    """SAMFormer-based channel-wise attention encoder for intra-visit interaction modeling (IIM)."""

    def __init__(self, num_channels, seq_len, hid_dim, use_revin=True):
        super().__init__()
        self.revin = RevIN(num_features=num_channels)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, hid_dim)
        self.transform = nn.Linear(seq_len, hid_dim)
        self.use_revin = use_revin

    def forward(self, x):
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2)  # (n, D, L)
        else:
            x_norm = x
        x_init = self.transform(x_norm)
        queries = self.compute_queries(x_norm)  # (n, D, hid_dim)
        keys = self.compute_keys(x_norm)
        values = self.compute_values(x_norm)
        if hasattr(F, 'scaled_dot_product_attention'):
            att_score = F.scaled_dot_product_attention(queries, keys, values)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values)
        out = x_init + att_score
        return out