"""Fixed-count pruning derived from the archived CLS cosine experiments.

This implementation intentionally fixes threshold-tie and batch-shape issues;
it is not a bit-for-bit reproduction of every historical notebook.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def prune_tokens(tokens, drop_rate=0.0, drop_most_similar=True):
    """Keep CLS and a fixed number of patches, preserving their original order.

    Args:
        tokens: Tensor with shape [batch, 1 + patches, embedding dimension].
        drop_rate: Fraction of patch tokens to remove, in [0, 1).
        drop_most_similar: Remove largest cosine scores when True.
    """
    if not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
        raise ValueError("drop_rate must be finite and in [0, 1)")
    if tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError("tokens must have shape [batch, 1 + patches, channels]")
    patches = tokens.shape[1] - 1
    count = int(patches * drop_rate)
    if count == 0:
        return tokens
    scores = F.cosine_similarity(tokens[:, 1:], tokens[:, :1], dim=-1)
    # Stable sorting makes tied scores deterministic and safe for batching.
    order = torch.argsort(scores, dim=1, descending=not drop_most_similar, stable=True)
    indices = order[:, :patches - count].sort(dim=1).values + 1
    selected = tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return torch.cat((tokens[:, :1], selected), dim=1)


class SimilarityPrunedViT(nn.Module):
    """Wrap a standard timm ViT with one CLS token and token pooling.

    Prune after positional embedding and/or after zero-indexed blocks.
    Each rate applies to patches remaining at that stage.
    """

    def __init__(self, model, embedding_drop=0.0, block_drop=0.0, blocks=(),
                 drop_most_similar=True):
        super().__init__()
        if getattr(model, "num_prefix_tokens", None) != 1:
            raise ValueError("Only a single CLS prefix token is supported")
        if getattr(model, "global_pool", None) != "token":
            raise ValueError("Only CLS/token pooling is supported")
        for rate in (embedding_drop, block_drop):
            if not math.isfinite(rate) or not 0 <= rate < 1:
                raise ValueError("Drop rates must be finite and in [0, 1)")
        if any(i < 0 or i >= len(model.blocks) for i in blocks):
            raise ValueError("Block index outside model depth")
        self.model = model
        self.embedding_drop = embedding_drop
        self.block_drop = block_drop
        self.blocks = frozenset(blocks)
        self.drop_most_similar = drop_most_similar

    def forward(self, images):
        model = self.model
        x = model._pos_embed(model.patch_embed(images))
        x = model.norm_pre(model.patch_drop(x))
        x = prune_tokens(x, self.embedding_drop, self.drop_most_similar)
        for index, block in enumerate(model.blocks):
            x = block(x)
            if index in self.blocks:
                x = prune_tokens(x, self.block_drop, self.drop_most_similar)
        return model.forward_head(model.norm(x))


def attention_prune(tokens, scores, drop_rate):
    """Keep highest-attention patches; CLS is always preserved."""
    if not math.isfinite(drop_rate) or not 0 <= drop_rate < 1:
        raise ValueError('drop_rate must be finite and in [0, 1)')
    if scores.shape != (tokens.shape[0], tokens.shape[1] - 1):
        raise ValueError('Expected one attention score per patch and batch item')
    keep = scores.shape[1] - int(scores.shape[1] * drop_rate)
    indices = torch.argsort(scores, descending=True, stable=True, dim=1)[:, :keep]
    indices = indices.sort(dim=1).values + 1
    return torch.cat((tokens[:, :1], tokens.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))), dim=1)


class CSAPViT(SimilarityPrunedViT):
    """Sim-Trim before the encoder, ATS after selected zero-indexed blocks.

    Recomputes the selected block's CLS attention row from normalized Q/K.
    This reference implementation is not an FPGA latency benchmark.
    """

    def forward(self, images):
        model = self.model
        x = model._pos_embed(model.patch_embed(images))
        x = model.norm_pre(model.patch_drop(x))
        x = prune_tokens(x, self.embedding_drop, self.drop_most_similar)
        for index, block in enumerate(model.blocks):
            scores = None
            if index in self.blocks and self.block_drop > 0:
                normalized = block.norm1(x)
                batch, count, channels = normalized.shape
                attn = block.attn
                qkv = attn.qkv(normalized).reshape(
                    batch, count, 3, attn.num_heads, channels // attn.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
                cls_attention = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
                scores = cls_attention.mean(dim=1)[:, 0, 1:]
            x = block(x)
            if scores is not None:
                x = attention_prune(x, scores, self.block_drop)
        return model.forward_head(model.norm(x))
