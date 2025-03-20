import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from timm.models.layers import trunc_normal_

from torch.nn import functional as F
class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = False,
        qkv_bias: bool = False,
        linear_bias: bool = True,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features, bias=linear_bias)
        self.bn = (
            nn.BatchNorm1d(dim, affine=False, eps=1e-6)
            if use_batch_norm
            else nn.Identity()
        )

        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls #averages over heads i think

        out = self.linear(x_cls)
        return out

AttentionPoolClassifier = AttentionPoolingClassifier


# TODO this is not completed
class AttentionPoolClassifierJepa(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 1,
        lin_bn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features)
        self.bn =  nn.BatchNorm1d(dim, affine=False, eps=1e-6) if lin_bn else nn.Identity()

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)
        return out#, x_cls