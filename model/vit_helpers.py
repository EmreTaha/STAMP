from timm.models.layers import to_3tuple, to_2tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=(16,16,8),
            in_chans=1,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.img_size = to_3tuple(img_size)
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)        
        
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _,  D, H, W = x.shape            
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)   # BCHWD -> BNC
        x = self.norm(x)
        return x
    
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width and depth
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    #TODO check where d comes
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = np.arange(h, dtype=float)
    grid_w = np.arange(w, dtype=float)
    grid_d = np.arange(d, dtype=float)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first #TODO xy in np, if swithc to pytorch, need to switch to yx #TODO check ij of pytorch
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, h, w, d])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0) # CLS token has pos embedding of 0s
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use 1/3rd to encode grid_h, grid_w, grid_d
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*D, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*D, D/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*D, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1) # (H*W*D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    h,w = grid_size
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class TimestepEmbedder(nn.Module):
    """
    Stolen from sit/dit
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=100):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=100):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb


torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.') 

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        if is_torch2:
            self.attn_drop  = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_future, temporal_relative_position=None):
        # TODO add that c and c future have the same shape
        B, N, C = x.shape
        B_future, N_future, C_future = x_future.shape
        # B1C -> B1H(C/H) -> BH1(C/H) #TODO updated
        q = self.wq(x_future).reshape(B_future, N_future, self.num_heads, C_future // self.num_heads).permute(0, 2, 1, 3) 
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if is_torch2 and not temporal_relative_position:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
            x_out = attn.transpose(1, 2).reshape(B_future, N_future, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_out = (attn @ v).transpose(1, 2).reshape(B_future, N_future, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C #TODO updated

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out
    

