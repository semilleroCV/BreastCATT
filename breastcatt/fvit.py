# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

from timm.models.vision_transformer import PatchEmbed
from timm.layers import AttentionPoolLatent

import timm.models.vision_transformer

import torch.nn as nn
import torch

from typing import Optional


class MultiModalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, cross_attn=False, cross_num_heads=8, norm_layer=nn.LayerNorm, fusion_alpha=1.0):
        super().__init__()
        # Reuse timm's Block for self-attention and mlp.
        self.block = timm.models.vision_transformer.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        self.use_cross_attn = cross_attn
        if cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=cross_num_heads)
        self.fusion_alpha = fusion_alpha

    def forward(self, x, text_embedding=None):
        # x: (B, L, D); text_embedding: (B, 1, D) or None
        x = self.block(x)
        if self.use_cross_attn and text_embedding is not None:
            q = text_embedding.transpose(0, 1)  # (1, B, D)
            k = x.transpose(0, 1)              # (L, B, D)
            v = x.transpose(0, 1)
            attn_output, _ = self.cross_attn(q, k, v)
            attn_output = attn_output.transpose(0, 1)  # (B, 1, D)
            x = x + self.fusion_alpha * attn_output.expand(-1, x.size(1), -1)
            text_embedding = text_embedding + attn_output
        return x, text_embedding


# New multi-modal VisionTransformer for finetuning classification.
class MultiModalVisionTransformer(nn.Module):
    def __init__(self, embed_dim, use_cross_attn=True, num_heads=16, in_chans=3,
                 num_classes: int = 1000, cross_num_heads=8, pos_drop_rate: float = 0.,
                 fusion_alpha=1.0, global_pool=False, depth=8, final_norm: bool = True,
                 drop_rate: float = 0., fc_norm: Optional[bool] = None,
                 **kwargs):
        # super().__init__(global_pool=global_pool, **kwargs)
        super().__init__()

        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.global_pool = global_pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # Replace original transformer blocks with MultiModalBlocks.
        self.blocks = nn.ModuleList([
            MultiModalBlock(embed_dim, num_heads, kwargs.get('mlp_ratio', 4),
                            cross_attn=use_cross_attn, cross_num_heads=cross_num_heads,
                            norm_layer=kwargs.get('norm_layer', nn.LayerNorm), fusion_alpha=fusion_alpha)
            for _ in range(depth)
        ])

        self.norm = kwargs.get('norm_layer', nn.LayerNorm)(embed_dim)

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=kwargs.get('mlp_ratio', 4),
                norm_layer=kwargs.get('norm_layer', nn.LayerNorm),
            )
        else:
            self.attn_pool = None
        self.fc_norm = kwargs.get('norm_layer', nn.LayerNorm)(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, text_embedding=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x, text_embedding = block(x, text_embedding)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x, text_embedding=None):
        features = self.forward_features(x, text_embedding)
        x = self.head(features)
        return x


# Factory function for multi-modal Vision Transformer (base version).
def multimodal_vit_base_patch16(**kwargs):
    model = MultiModalVisionTransformer(
        patch_size=16, in_chans=1, embed_dim=1024, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        cross_num_heads=8, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=False, **kwargs)
    return model

# Factory function for multi-modal Vision Transformer (large version).
def multimodal_vit_large_patch16(**kwargs):
    model = MultiModalVisionTransformer(
        patch_size=16, in_chans=1, embed_dim=1024, depth=12, num_heads=16, mlp_ratio=4, qkv_bias=True,
        cross_num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=False, **kwargs)
    return model

# Factory function for multi-modal Vision Transformer (huge version).
def multimodal_vit_huge_patch16(**kwargs):
    model = MultiModalVisionTransformer(
        patch_size=16, in_chans=1, embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4, qkv_bias=True,
        cross_num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=False, **kwargs)
    return model