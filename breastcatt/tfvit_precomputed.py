# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

from timm.layers.patch_embed import PatchEmbed
import timm.models.vision_transformer

import torch.nn as nn
import torch
from typing import Optional
from dataclasses import dataclass
import os
import json
from huggingface_hub import hf_hub_download
import urllib.request
from mae.pos_embed import get_2d_sincos_pos_embed

@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None

def download_mae_weights(model_size="base", force_download=False):
    """
    Download MAE pretrained weights from Facebook's servers.
    
    Args:
        model_size (str): Size of the model ("base" or "large")
        force_download (bool): Whether to force re-download if file exists
        
    Returns:
        str: Path to the downloaded checkpoint file
    """
    # Define download URLs for different model sizes
    mae_urls = {
        "base": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "large": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"
    }
    
    if model_size not in mae_urls:
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(mae_urls.keys())}")
    
    # Create checkpoints directory
    checkpoint_dir = "checkpoints/fvit"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define local file path
    filename = f"mae_pretrain_vit_{model_size}.pth"
    local_path = os.path.join(checkpoint_dir, filename)
    
    # Check if file already exists
    if os.path.exists(local_path) and not force_download:
        print(f"✅ MAE {model_size} weights already exist at: {local_path}")
        return local_path
    
    # Download the file
    url = mae_urls[model_size]
    print(f"📥 Downloading MAE {model_size} weights from Facebook's servers...")
    print(f"URL: {url}")
    print(f"Saving to: {local_path}")
    
    try:
        # Download with progress indication
        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, block_num * block_size * 100.0 / total_size)
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, local_path, download_progress)
        print(f"\n✅ Successfully downloaded MAE {model_size} weights to: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"\n❌ Error downloading MAE weights: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)  # Clean up partial download
        raise
    
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

class MultiModalVisionTransformer(nn.Module):
    def __init__(self, embed_dim, use_cross_attn=True, use_segmentation=False, num_heads=16, in_chans=3,
                 num_classes: int = 1000, cross_num_heads=8, pos_drop_rate: float = 0.,
                 fusion_alpha=1.0, depth=8, drop_rate: float = 0., **kwargs):
        super().__init__()

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # Language model for text embedding (only if cross-attention is enabled)
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            # GatorTron's output is 1024, project it to the ViT's embed_dim
            self.text_embed_proj = nn.Sequential(
                nn.Linear(1024, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # Segmentation ROI (only if enabled)
        self.use_segmentation = use_segmentation

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MultiModalBlock(embed_dim, num_heads, kwargs.get('mlp_ratio', 4),
                            cross_attn=use_cross_attn, cross_num_heads=cross_num_heads,
                            norm_layer=kwargs.get('norm_layer', nn.LayerNorm), fusion_alpha=fusion_alpha)
            for _ in range(depth)
        ])

        self.norm = kwargs.get('norm_layer', nn.LayerNorm)(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.num_classes = num_classes

        self.initialize_weights()

    @classmethod
    def create_from_init_args(cls, init_args, checkpoint_path: Optional[str] = None, map_location="cpu"):
        """
        Create a model from init_args, optionally loading weights from a checkpoint.
        
        Args:
            init_args: Dictionary of initialization arguments
            checkpoint_path: Path to a checkpoint to load weights from
            map_location: Device to map the checkpoint to
        
        Returns:
            A MultiModalVisionTransformer model
        """

        if checkpoint_path: # load weights from MAE
            return cls.from_pretrained(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                **init_args
            )
        return cls(**init_args)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, text_embedding=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.use_cross_attn and text_embedding is not None and self.text_embed_proj is not None:
            text_embedding = self.text_embed_proj(text_embedding).unsqueeze(1) # (B, 1, embed_dim)

        for block in self.blocks:
            x, text_embedding = block(x, text_embedding)
        x = self.norm(x)
        outcome = x[:, 0]
        return outcome

    def forward_loss(self, pixel_values, labels=None, text_embedding=None, segmentation_mask=None):
        # Apply segmentation mask only if enabled
        if self.use_segmentation and segmentation_mask is not None:
            pixel_values = pixel_values * segmentation_mask

        # The text_embedding is now expected to be a tensor from the batch
        outcome = self.forward_features(pixel_values, text_embedding)
        logits = self.head_drop(outcome)
        logits = self.head(logits)
        loss = None
        if labels is not None:
            if self.num_classes == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels.unsqueeze(1).float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
        return ModelOutput(logits=logits, loss=loss)

    def forward(self, pixel_values, labels=None, text_embedding=None, segmentation_mask=None):
        return self.forward_loss(pixel_values, labels=labels,
                                 text_embedding=text_embedding,
                                 segmentation_mask=segmentation_mask)

    def save_pretrained(self, save_directory):
        """
        Save the model weights and configuration to the specified directory in Hugging Face style.
        Extra keyword arguments are ignored for compatibility.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Save config with safe defaults
        config = {
            "embed_dim": getattr(self.patch_embed.proj, 'out_channels', 768),
            "use_cross_attn": self.use_cross_attn,
            "use_segmentation": self.use_segmentation,
            "num_heads": 12,  # Default value, can be customized
            "in_chans": getattr(self.patch_embed.proj, 'in_channels', 1),
            "num_classes": self.num_classes,
            "cross_num_heads": 8,  # Default value
            "fusion_alpha": 1.0,  # Default value
            "depth": len(self.blocks),
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, checkpoint_path, map_location, **model_kwargs):
        # 1. Initialize the multimodal model
        model = cls(**model_kwargs)

        # 2. Load MAE checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        mae_state = torch.load(checkpoint_path, map_location=map_location)

        if "model" in mae_state:
            mae_state = mae_state["model"]  # in case it's nested

        new_state_dict = {}
        missing = []
        loaded = []
        if "patch_embed.proj.weight" in mae_state:
            weight = mae_state["patch_embed.proj.weight"]
            if weight.shape[1] == 3:  
                print("Adapting patch_embed.proj.weight from 3 channels to 1 channel...")
                mae_state["patch_embed.proj.weight"] = weight.mean(dim=1, keepdim=True)  # [768, 1, 16, 16]
        # 3. Remap MAE layer names to multimodal model
        for k, v in mae_state.items():
            if k.startswith("patch_embed.") or k.startswith("norm."):
                new_state_dict[k] = v
                loaded.append(k)
            elif k.startswith("blocks."):
                # Convert: blocks.0.attn.qkv.weight → blocks.0.block.attn.qkv.weight
                parts = k.split(".")
                block_id = parts[1]
                rest = ".".join(parts[2:])
                new_k = f"blocks.{block_id}.block.{rest}"
                if new_k in model.state_dict():
                    new_state_dict[new_k] = v
                    loaded.append(new_k)
                else:
                    missing.append(new_k)

        # 4. Load weights into the model
        msg = model.load_state_dict(new_state_dict, strict=False)

        print(f"\n✅ Loaded weights: {len(loaded)} layers.")
        print(f"❌ Not found in model: {len(missing)} layers.")
        print(f"📋 Details of load_state_dict:\n{msg}")

        return model


# Factory function for multi-modal Vision Transformer (small version).
def multimodal_vit_small_patch16(**kwargs):
    """
    Small ViT: Configure model parameters through kwargs.
    
    Key parameters (all optional with defaults):
    - use_cross_attn: Whether to use cross-attention (default: True)
    - use_segmentation: Whether to use segmentation (default: False)
    - num_classes: Number of output classes (default: 1)
    - checkpoint_path: Path to pretrained weights (default: None, MAE small not available)
    - map_location: Device to map checkpoint to (default: "cpu")
    - embed_dim: Embedding dimension (default: 384)
    - num_heads: Number of attention heads (default: 6)
    - in_chans: Number of input channels (default: 1)
    - cross_num_heads: Number of cross-attention heads (default: 4)
    - fusion_alpha: Fusion alpha parameter (default: 1.0)
    - depth: Transformer depth (default: 4)
    """
    # Extract parameters from kwargs with defaults
    use_cross_attn = kwargs.get('use_cross_attn', True)
    use_segmentation = kwargs.get('use_segmentation', False)
    num_classes = kwargs.get('num_classes', 2)
    embed_dim = kwargs.get('embed_dim', 384)
    num_heads = kwargs.get('num_heads', 6)
    in_chans = kwargs.get('in_chans', 1)
    cross_num_heads = kwargs.get('cross_num_heads', 4)
    fusion_alpha = kwargs.get('fusion_alpha', 1.0)
    depth = kwargs.get('depth', 4)
    
    # Handle checkpoint path
    checkpoint_path = kwargs.get('checkpoint_path', None)
    map_location = kwargs.get('map_location', "cpu")

    # Create a clean init_args dict
    init_args = dict(
        patch_size=16,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        cross_num_heads=cross_num_heads,
        fusion_alpha=fusion_alpha,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_cross_attn=use_cross_attn,
        use_segmentation=use_segmentation,
        num_classes=num_classes
    )

    # Add any remaining kwargs
    for k, v in kwargs.items():
        if k not in init_args and k not in ['checkpoint_path', 'map_location']:
            init_args[k] = v

    return MultiModalVisionTransformer.create_from_init_args(init_args, checkpoint_path, map_location)

# Factory function for multi-modal Vision Transformer (base version).
def multimodal_vit_base_patch16(**kwargs):
    """
    Base ViT: Configure model parameters through kwargs.
    
    Key parameters (all optional with defaults):
    - use_cross_attn: Whether to use cross-attention (default: True)
    - use_segmentation: Whether to use segmentation (default: False)
    - num_classes: Number of output classes (default: 1)
    - checkpoint_path: Path to pretrained weights (default: auto-download MAE base)
    - map_location: Device to map checkpoint to (default: "cpu")
    - embed_dim: Embedding dimension (default: 768)
    - num_heads: Number of attention heads (default: 12)
    - in_chans: Number of input channels (default: 1)
    - cross_num_heads: Number of cross-attention heads (default: 8)
    - fusion_alpha: Fusion alpha parameter (default: 1.0)
    - depth: Transformer depth (default: 12)
    """
    # Extract parameters from kwargs with defaults, but don't remove them from kwargs
    use_cross_attn = kwargs.get('use_cross_attn', True)
    use_segmentation = kwargs.get('use_segmentation', False)
    num_classes = kwargs.get('num_classes', 2)
    embed_dim = kwargs.get('embed_dim', 768)
    num_heads = kwargs.get('num_heads', 12)
    in_chans = kwargs.get('in_chans', 1)
    cross_num_heads = kwargs.get('cross_num_heads', 8)
    fusion_alpha = kwargs.get('fusion_alpha', 1.0)
    depth = kwargs.get('depth', 12)
    
    # Handle checkpoint path - auto-download if not provided
    checkpoint_path = kwargs.get('checkpoint_path', None)
    if checkpoint_path is None:
        checkpoint_path = download_mae_weights("base")
    map_location = kwargs.get('map_location', "cpu")

    # Create a clean init_args dict with all parameters
    init_args = dict(
        patch_size=16,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        cross_num_heads=cross_num_heads,
        fusion_alpha=fusion_alpha,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_cross_attn=use_cross_attn,
        use_segmentation=use_segmentation,
        num_classes=num_classes
    )
    
    # Add any remaining kwargs to handle additional parameters
    # but exclude checkpoint_path and map_location to avoid duplicates
    for k, v in kwargs.items():
        if k not in init_args and k not in ['checkpoint_path', 'map_location']:
            init_args[k] = v

    # Create model from init_args and checkpoint if provided
    model = MultiModalVisionTransformer.create_from_init_args(init_args, checkpoint_path, map_location)
    
    return model

# Factory function for multi-modal Vision Transformer (large version).
def multimodal_vit_large_patch16(**kwargs):
    """
    Large ViT: Configure model parameters through kwargs.
    
    Key parameters (all optional with defaults):
    - use_cross_attn: Whether to use cross-attention (default: True)
    - use_segmentation: Whether to use segmentation (default: False)
    - num_classes: Number of output classes (default: 1)
    - checkpoint_path: Path to pretrained weights (default: auto-download MAE large)
    - map_location: Device to map checkpoint to (default: "cpu")
    - embed_dim: Embedding dimension (default: 1024)
    - num_heads: Number of attention heads (default: 16)
    - in_chans: Number of input channels (default: 1)
    - cross_num_heads: Number of cross-attention heads (default: 16)
    - fusion_alpha: Fusion alpha parameter (default: 1.0)
    - depth: Transformer depth (default: 24)
    """
    # Extract parameters from kwargs with defaults
    use_cross_attn = kwargs.get('use_cross_attn', True)
    use_segmentation = kwargs.get('use_segmentation', False)
    num_classes = kwargs.get('num_classes', 2)
    embed_dim = kwargs.get('embed_dim', 1024)
    num_heads = kwargs.get('num_heads', 16)
    in_chans = kwargs.get('in_chans', 1)
    cross_num_heads = kwargs.get('cross_num_heads', 16)
    fusion_alpha = kwargs.get('fusion_alpha', 1.0)
    depth = kwargs.get('depth', 24)

    # Handle checkpoint path - auto-download if not provided
    checkpoint_path = kwargs.get('checkpoint_path', None)
    if checkpoint_path is None:
        checkpoint_path = download_mae_weights("large")
    map_location = kwargs.get('map_location', "cpu")
    
    init_args = dict(
        patch_size=16,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        cross_num_heads=cross_num_heads,
        fusion_alpha=fusion_alpha,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_cross_attn=use_cross_attn,
        use_segmentation=use_segmentation,
        num_classes=num_classes
    )

    # Add any remaining kwargs
    for k, v in kwargs.items():
        if k not in init_args and k not in ['checkpoint_path', 'map_location']:
            init_args[k] = v

    return MultiModalVisionTransformer.create_from_init_args(init_args, checkpoint_path, map_location)

