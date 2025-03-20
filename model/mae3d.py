from functools import partial

import numpy as np
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple, to_3tuple
from timm.models.layers import trunc_normal_

from .vit_helpers import get_3d_sincos_pos_embed, PatchEmbed3D

# Shamelessly stolen from facebook mae

# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


class MaskedAutoencoderViT3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224,224,32), patch_size=(16,16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.grid_shape = (img_size[0]//self.patch_size[0], img_size[1]//self.patch_size[1], img_size[2]//self.patch_size[2])

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, np.prod(self.patch_size) * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # TODO Fix this
        #@torch.jit.ignore
        #def no_weight_decay(self):
        #    return {'pos_embed', 'cls_token'}

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_shape, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_shape, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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
        elif isinstance(m, nn.Conv3d):
            # TODO this is from jepa, they do it for lin as well
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L,  p1*p2*p3*C)
        """
        p1, p2, p3 = self.patch_size
        assert imgs.shape[-3] % p1 == 0 and imgs.shape[-2] % p2 == 0 and imgs.shape[-1] % p3 == 0

        h = imgs.shape[-3] // p1
        w = imgs.shape[-2] // p2
        d = imgs.shape[-1] // p3 #TODO if d is last
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p1, w, p2, d, p3))
        x = torch.einsum('nchpwqdb->nhwdpqbc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, np.prod(self.patch_size) * self.in_chans))
        return x

    def unpatchify3D(self, x):
        """
        x: (N, L, p1*p2*p3 *C)
        imgs: (N, C, H, W, D)
        """
        p1, p2, p3 = self.patch_size
        h, w, d = self.grid_shape
        assert h * w * d == x.shape[1] # number of patches
        
        x = x.reshape(shape=(x.shape[0], h, w, d, p1, p2, p3, self.in_chans))
        x = torch.einsum('nhwdpqbc->nchpwqdb', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p1, w * p2, d * p3))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W, D]
        pred: [N, L, p1*p2*p3*1]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify3D(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae3d_vit_small_patch16_dec192d4b(**kwargs):
    """ ViT-Small (ViT-S/16) with smaller decoder
    """
    model = MaskedAutoencoderViT3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae3d_vit_small_patch16_dec384d8b(patch_size=(16,16,8), **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model = MaskedAutoencoderViT3D(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae3d_vit_base_patch16_dec384d8b(patch_size=(16,16,8), **kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae3d_vit_base_patch16_dec768d8b(patch_size=(16,16,8),**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae3d_vit_large_patch16_dec384d8b(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=16, embed_dim=960, depth=24, num_heads=16,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae3d_vit_huge_patch14_dec384d8b(**kwargs):
    model = MaskedAutoencoderViT3D(
        patch_size=14, embed_dim=1200, depth=32, num_heads=16,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae3d_vit_small_patch16_d = mae3d_vit_small_patch16_dec192d4b # decoder: 192 dim, 4 blocks, 8 heads
mae3d_vit_small_patch16 = mae3d_vit_small_patch16_dec384d8b # decoder: 192 dim, 8 blocks, 16 heads
mae3d_vit_base_patch16 = mae3d_vit_base_patch16_dec384d8b  # decoder: 384 dim, 8 blocks, 16 heads
mae3d_vit_base_patch16_ld = mae3d_vit_base_patch16_dec768d8b  # decoder: 384 dim, 8 blocks, 16 heads
mae3d_vit_large_patch16 = mae3d_vit_large_patch16_dec384d8b  # decoder: 512 dim, 8 blocks, 16 heads
mae3d_vit_huge_patch14 = mae3d_vit_huge_patch14_dec384d8b  # decoder: 512 dim, 8 blocks, 16 heads