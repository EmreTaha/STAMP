from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, Attention, LayerScale
from timm.layers import Mlp, DropPath

from typing import Optional

from .vit_helpers import get_3d_sincos_pos_embed, PatchEmbed3D
from timm.models.layers import to_3tuple
import torch.distributions as td
from torch.distributions.kl import  register_kl, _kl_categorical_categorical

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.')
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

@register_kl(td.RelaxedOneHotCategorical, td.RelaxedOneHotCategorical)
def _kl_relaxedonehotcategorical_relaxedonehotcategorical(p, q):
    return _kl_categorical_categorical(p.base_dist._categorical, q.base_dist._categorical)

class CrossSelfDecoderBlock(nn.Module):
    # In original MAE, there is no drop path etc so they are omitted here
    # NOTE This follows shared norm-CA-qnorm-SA-qnorm
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,

    ) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_norm, attn_drop,proj_drop,norm_layer)
        self.attention = Attention(
                                    dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_norm=qk_norm,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    norm_layer=norm_layer,
                                )
        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.norm_3 = norm_layer(dim) 
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop_path)

    def forward(self, x, x_future, src_mask=None):
        # Following timm, first norm
        x_conc = self.norm_1(torch.cat((x_future, x), dim=1))
        x_future, x = torch.split(x_conc, [x_future.shape[1],x.shape[1]], dim=1)

        x_cross = x_future + self.cross_attention( x, x_future, src_mask)

        # Following timm, first norm
        x_cross = x_cross + self.attention(self.norm_2(x_cross))

        # Following timm, first norm
        x_cross = x_cross + self.mlp(self.norm_3(x_cross))
        return x_cross


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
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
            self.attn_dropout = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_future, src_mask=None):

        B, N, C = x.shape
        N_f = x_future.shape[1]
        # B1C -> B1H(C/H) -> BH1(C/H) #TODO updated
        q = self.wq(x_future).reshape(B, N_f, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if is_torch2 and not src_mask:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
            x_out = attn.transpose(1, 2).reshape(B, N_f, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

            # Introduced in RSP
            if src_mask is not None:
                src_mask = src_mask.unsqueeze(1) # shape (B, 1, N, N)?
                src_mask = src_mask.unsqueeze(1) # shape (B, 1, 1, N, N)?
                src_mask = src_mask.repeat(1, self.num_heads, N, 1) #
                attn = attn.masked_fill(src_mask == 0, -1e4)

            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            x_out = (attn @ v).transpose(1, 2).reshape(B, N_f, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C #TODO updated

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class RSP3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224,224,32), patch_size=(16,16,16), in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, kl_scale=0.01, 
                 kl_balance=0.2, kl_freebit=0.1, perturb=0.5, stoch=32, discrete=32,
                 mask_all=False, prior_dist='straight'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_prefix_tokens = 1 # There is always a cls token, but this is needed for registers

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.grid_shape = (img_size[0]//self.patch_size[0], img_size[1]//self.patch_size[1], img_size[2]//self.patch_size[2])

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # Determine the size of stochastic variable
        # For discrete latents:
        # - It consists of M N-dimensional one-hot vectors (M: stoch, N: discrete)
        # For continuous latents:
        # - It is M-dimenisonal gaussian. Thus it has M * 2 for mean and std
        stoch_size = stoch * discrete if discrete != 0 else stoch * 2

        # Posterior takes both src_h and tgt_h
        # Thus it has embed_dim * 2 as an input dimension
        self.to_posterior = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        # Prior only takes src_h
        # Thus it has embed_dim as an input dimension
        self.to_prior = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, stoch_size),
        )

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_deter = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_stoch = nn.Linear(stoch_size, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            CrossSelfDecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, np.prod(self.patch_size) * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.mask_all = mask_all # If True, mask all the tokens for the variational forwarding
        self.norm_pix_loss = norm_pix_loss
        self.kl_scale = kl_scale
        self.kl_balance = kl_balance
        self.kl_freebit = kl_freebit
        self.noise_scale = perturb
        self.stoch = stoch
        self.discrete = discrete
        self.prior_dist = prior_dist

        self.initialize_weights()

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
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

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

        return:
            x_masked: [N, L, D], masked sequence
            mask: [N, L], binary mask
            ids_restore: [N, L], restore indices
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
    
    def perturb(self, x):
        """
        From RSP, add noise to the input
        """
        noise = torch.randn_like(x) * self.noise_scale
        
        return x + noise

    def forward_encoder(self, x, mask_ratio=0.0):
        """
        To make compatible with RSP, it forwards once at a time
        """

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token and register tokens
        x = x + self.pos_embed[:, self.num_prefix_tokens:, :]

        # mask only future: length -> length * mask_ratio
        if mask_ratio != 0.0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls and register tokens
        cls_token = self.cls_token + self.pos_embed[:, :self.num_prefix_tokens, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder_fut(self, h,z):
        """
        Predict future from h_t + z_t+k. For querry it only uses the mask tokens
        """
        # embed tokens
        kvx_h = self.get_feat(h, z) # decoder_pos_embed is added to the deterministic part inside the function

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x_future = mask_tokens + self.decoder_pos_embed
        # x_future has only mask tokens

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_future = blk(kvx_h, x_future)
        x_future = self.decoder_norm(x_future)

        # predictor projection
        x_future = self.decoder_pred(x_future)

        # remove cls/register token
        x_future = x_future[:, self.num_prefix_tokens:, :]

        return x_future

    def forward_decoder_mae(self, h, ids_restore, return_reg=False):
        #ids_restore comes from x_future only

        # embed tokens
        h = self.decoder_embed(h)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(h.shape[0], ids_restore.shape[1] + 1 - h.shape[1], 1)
        h_ = torch.cat([h[:, self.num_prefix_tokens:, :], mask_tokens], dim=1)  # no cls/register token for future input
        h_ = torch.gather(h_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, h.shape[2]))  # unshuffle
        h = torch.cat([h[:, :self.num_prefix_tokens, :], h_], dim=1)  # append cls/register token to future input
        kvx_h = h + self.decoder_pos_embed
        # x is full shape + 1 cls.

        # get mask tokens
        mask_tokens = self.mask_token.repeat(h.shape[0], h.shape[1], 1)
        x = mask_tokens + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(kvx_h, x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

      # return register and cls tokens
        if return_reg:
            return x[:, :self.num_prefix_tokens, :]

        # remove cls/register token
        x = x[:, self.num_prefix_tokens:, :]

        return x

    def forward_loss(self, imgs, pred, mask=None):
        """
        imgs: [N, C, H, W, D]
        pred: [N, L, p1*p2*p3*C]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify3D(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        if mask is not None:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = loss.mean()

        return loss
    
    def get_feat(self, h, z):
        h = self.decoder_embed_deter(h) + self.decoder_pos_embed

        if self.discrete != 0:
            z = z.reshape(*z.shape[:-2], 1, self.stoch * self.discrete)
        z = self.decoder_embed_stoch(z)

        feat = torch.cat([z, h], dim=1)
        return feat
    
    def make_dist(self, logits):
        if self.discrete != 0:
            logits = logits.reshape([-1, self.stoch, self.discrete])
            if self.prior_dist == 'straight':
                dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
            elif self.prior_dist == 'soft':
                dist = td.Independent(td.RelaxedOneHotCategorical(temperature=torch.tensor([0.1]).to(device), logits=logits), 1)
        else:
            mean, std = torch.split(logits, 2, -1)
            dist = td.Normal(mean, std)
        return dist

    def kl_loss(self, post_logits, prior_logits):
        """
        Symmetric KL divergence loss
        """
        balance = self.kl_balance
        freebit = self.kl_freebit
        post_to_prior_kl = td.kl_divergence(
            self.make_dist(post_logits), self.make_dist(prior_logits.detach())
        )
        prior_to_post_kl = td.kl_divergence(
            self.make_dist(post_logits.detach()), self.make_dist(prior_logits)
        )
        kl_value = (
            post_to_prior_kl * balance + prior_to_post_kl * (1.0 - balance)
        ).mean()
        # Symmetric KL is not 2 KLs with different order, this is from dreamerv3
        kl_loss = torch.maximum(kl_value, torch.ones_like(kl_value) * freebit) # TODO what is this horse shit
        return kl_loss, kl_value, [post_to_prior_kl.detach().mean(), prior_to_post_kl.detach().mean()]

    @torch.no_grad()
    def forward_prior(self, imgs):
        """
        Sample from prior and generate
        """

        h, _, _ = self.forward_encoder(imgs, mask_ratio=0)
        prior_logits = self.to_prior(h[:, 0]) # Use CLS tokens
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample()

        pred = self.forward_decoder_fut(h, prior_z)
        return pred

    def forward(self, imgs, imgs_future, mask_ratio=0.75, return_reg=False, return_indv_loss=False):
        # If mask_all is False, Encode without masking for stochastic latent
        h, _, _ = self.forward_encoder(imgs, mask_ratio=0)
        h_future, _, _ = self.forward_encoder(self.perturb(imgs_future), mask_ratio=self.mask_all*mask_ratio) # False olunca posterior butun future'i gorerek olusturuluyor

        # Posterior distribution from both images
        post_h = torch.cat([h[:, 0], h_future[:, 0]], -1) # Use CLS tokens
        post_logits = self.to_posterior(post_h)
        post_dist = self.make_dist(post_logits)
        post_z = post_dist.rsample()

        # Prior distribution only from current images
        prior_h = h[:, 0] # Use CLS tokens
        prior_logits = self.to_prior(prior_h)
        #NOTE moved to forwad_prior
        #prior_dist = self.make_dist(prior_logits)
        #prior_z = prior_dist.rsample()

        # Stochastic loss
        future_pred = self.forward_decoder_fut(h, post_z)

        loss_post = self.forward_loss(imgs_future, future_pred) # Predict future fully, as if no masking
        kl_loss, kl_value, kl_indv = self.kl_loss(post_logits, prior_logits)

        # MAE loss for standard MAE except decoder querry has only mask tokens agains...
        latent_future, mask, ids_restore = self.forward_encoder(imgs_future, mask_ratio) # Forward with masking

        pred = self.forward_decoder_mae(latent_future, ids_restore, return_reg)  # [N, L, p*p*1]
        if return_reg: return pred # Return cls (all register tokens)
        mae_loss = self.forward_loss(imgs_future, pred, mask) # Predict the future
        loss = loss_post + self.kl_scale * kl_loss + mae_loss
        
        if return_indv_loss:
            return loss, future_pred, mask, [loss_post, kl_indv[0], kl_indv[1], mae_loss]

        return loss, future_pred, mask


def rsp_mae3d_vit_small_patch16_dec1926d4b(**kwargs):
    """ ViT-Small (ViT-S/16) with smaller decoder
    """
    model = RSP3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def rsp_mae3d_vit_small_patch16_dec384d8b( **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model = RSP3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def rsp_mae3d_vit_base_patch16_dec384d8b(patch_size=(16,16,8), **kwargs):
    model = RSP3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def rsp_mae3d_vit_base_patch16_dec528d8b(patch_size=(16,16,8), **kwargs):
    model = RSP3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def rsp_mae3d_vit_base_patch16_dec768d8b(patch_size=(16,16,8),**kwargs):
    model = RSP3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def rsp_mae3d_vit_large_patch16_dec384d8b(**kwargs):
    model = RSP3D(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def rsp_mae3d_vit_huge_patch14_dec384d8b(**kwargs):
    model = RSP3D(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
rsp_mae3d_vit_small_patch16_d = rsp_mae3d_vit_small_patch16_dec1926d4b # decoder: 192 dim, 4 blocks, 8 heads
rsp_mae3d_vit_small_patch16 = rsp_mae3d_vit_small_patch16_dec384d8b # decoder: 384 dim, 8 blocks, 16 heads
rsp_mae3d_vit_base_patch16 = rsp_mae3d_vit_base_patch16_dec384d8b  # decoder: 384 dim, 8 blocks
rsp_mae3d_vit_base_patch16_md = rsp_mae3d_vit_base_patch16_dec528d8b  # decoder: 528 dim, 8 blocks
rsp_mae3d_vit_base_patch16_ld = rsp_mae3d_vit_base_patch16_dec768d8b  # decoder: 768 dim, 8 blocks, 16 heads
rsp_mae3d_vit_large_patch16 = rsp_mae3d_vit_large_patch16_dec384d8b  # decoder: 512 dim, 8 blocks
rsp_mae3d_vit_huge_patch14 = rsp_mae3d_vit_huge_patch14_dec384d8b  # decoder: 512 dim, 8 blocks