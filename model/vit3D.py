import timm.models.vision_transformer
from functools import partial
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from timm.models.layers import to_3tuple
import math
from functools import partial, reduce
from operator import mul
from .vit_helpers import PatchEmbed3D, get_3d_sincos_pos_embed, TimestepEmbedder
import torch.distributions as td


class VisionTransformer3D(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,img_size=(224,224,32), patch_size=(16,16,16), in_chans=3, global_pool=False, contrastive_cls=False, turnoff_last_ln=False, use_learnable_pos_emb=True, time_embed=False, return_features=False, **kwargs):
        super(VisionTransformer3D, self).__init__(**kwargs)
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, kwargs['embed_dim'])
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, kwargs['embed_dim']) * .02)

        if not use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, kwargs['embed_dim']), requires_grad=False)
            self.grid_shape = (kwargs['img_size'][0]//self.patch_size[0], kwargs['img_size'][1]//self.patch_size[1], kwargs['img_size'][2]//self.patch_size[2]) 
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_shape, cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # In a simple setup, I found retfound style slightly better
        trunc_normal_(self.head.weight, std=2e-5) # Retfound style
        nn.init.normal_(self.cls_token, std=.02)
        #trunc_normal_(self.backbone.head.weight, std=0.01) # MOCOv3 style
        self.embed_dim = kwargs['embed_dim']

        self.return_features = return_features
        self.contrastive_cls = contrastive_cls # This is for if any contrastive loss is used for pretraining on the cls token
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            
            # Push the fc_norm inside the head and make the original norm an identity
            self.fc_norm = nn.Identity()
            self.head =  torch.nn.Sequential(norm_layer(self.embed_dim), #TODO fix this for contrastive cls
                                                 self.head)

            #del self.norm  # remove the original norm
            if not self.contrastive_cls: self.norm = nn.Identity()  # replace with identity # if clause is added 23.10.2024 # No need bc no contrastive cls skips this
        else:
            self.norm = kwargs['norm_layer'](self.embed_dim) #TODO add this as optional. this is beforre returning CLS

        if turnoff_last_ln:
            if isinstance(self.norm, nn.LayerNorm):
                self.norm = nn.Identity()
            # NOTE no need for head, contrastive part doesnt use it at all

        if time_embed:
            self.encoder_time_embed = TimestepEmbedder(kwargs['embed_dim'], 64, 160)
        else:
            self.encoder_time_embed = None

        # TODO Fix this
        #@torch.jit.ignore
        #def no_weight_decay(self):
        #    return {'pos_embed', 'cls_token'}

    def forward_features(self, x, delta_t=None):
        # Slightly different from the original timm implementation. It returns the cls token or average of all tokens as well
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.encoder_time_embed is not None: # add only to cls option now
            x[:,:self.num_prefix_tokens, :] = x[:,:self.num_prefix_tokens, :] + self.encoder_time_embed(delta_t).unsqueeze(1)

    
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool and self.contrastive_cls:
            x = self.norm(x) # NOTE this is added 23.10.2024, TC transformer pretrained models before, has ln, but loading it without contrastive cls will lead to loss of ln
            x_glob = x.mean(dim=1)
            outcome = self.fc_norm(x_glob) # I changed it to identity, real fc_norm is in the head
        elif self.global_pool:
            x_glob = x[:, self.num_prefix_tokens:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x_glob) # I changed it to identity, real fc_norm is in the head
        else:
            x = self.norm(x) # This is before the cls token
            if self.return_features:
                return x
            outcome = x[:, 0]

        return outcome
    
    @torch.no_grad()
    def get_last_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        return x
    
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        # Clumped down a lot, original mae uses old timm, latest timm is too new
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor = None) -> torch.Tensor:
        x = self.forward_features(x, delta_t)
        x = self.forward_head(x)
        return x
    

class stoch_VisionTransformer3D(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,img_size=(224,224,32), patch_size=(16,16,16), in_chans=3, global_pool=False, contrastive_cls=False, turnoff_last_ln=False, use_learnable_pos_emb=True, time_embed=False, return_features=False, n_sample=1, **kwargs):
        super(stoch_VisionTransformer3D, self).__init__(**kwargs)
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, kwargs['embed_dim'])
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, kwargs['embed_dim']) * .02)

        if not use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, kwargs['embed_dim']), requires_grad=False)
            self.grid_shape = (kwargs['img_size'][0]//self.patch_size[0], kwargs['img_size'][1]//self.patch_size[1], kwargs['img_size'][2]//self.patch_size[2]) 
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_shape, cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # In a simple setup, I found retfound style slightly better
        trunc_normal_(self.head.weight, std=2e-5) # Retfound style
        nn.init.normal_(self.cls_token, std=.02)
        #trunc_normal_(self.backbone.head.weight, std=0.01) # MOCOv3 style
        self.embed_dim = kwargs['embed_dim']

        self.return_features = return_features
        self.contrastive_cls = contrastive_cls # This is for if any contrastive loss is used for pretraining on the cls token
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            
            # Push the fc_norm inside the head and make the original norm an identity
            self.fc_norm = nn.Identity()
            self.head =  torch.nn.Sequential(norm_layer(self.embed_dim), #TODO fix this for contrastive cls
                                                 self.head)

            #del self.norm  # remove the original norm
            if not self.contrastive_cls: self.norm = nn.Identity()  # replace with identity # if clause is added 23.10.2024 # No need bc no contrastive cls skips this
        else:
            self.norm = kwargs['norm_layer'](self.embed_dim) #TODO add this as optional. this is beforre returning CLS

        if turnoff_last_ln:
            if isinstance(self.norm, nn.LayerNorm):
                self.norm = nn.Identity()

        if time_embed:
            self.encoder_time_embed = TimestepEmbedder(kwargs['embed_dim'], 64, 160)
        else:
            self.encoder_time_embed = None

        self.n_sample = n_sample
        self.stoch = 32
        self.discrete = 32
        self.stoch_size = self.stoch * self.discrete

        self.to_prior = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.SiLU(),# NOTE 19.01.2025 nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.stoch_size),
        )

        self.decoder_embed = nn.Linear(self.embed_dim, 384, bias=True)
        self.decoder_embed_stoch = nn.Linear(self.stoch_size, 384, bias=True)

    def forward_features(self, x, delta_t=None):
        # Slightly different from the original timm implementation. It returns the cls token or average of all tokens as well
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.encoder_time_embed is not None: # add only to cls option now
            x[:,:self.num_prefix_tokens, :] = x[:,:self.num_prefix_tokens, :] + self.encoder_time_embed(delta_t).unsqueeze(1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool and self.contrastive_cls:
            x = self.norm(x) # NOTE this is added 23.10.2024, TC transformer pretrained models before, has ln, but loading it without contrastive cls will lead to loss of ln
            x_glob = x.mean(dim=1)
            outcome = self.fc_norm(x_glob) # I changed it to identity, real fc_norm is in the head
        elif self.global_pool:
            x_glob = x[:, self.num_prefix_tokens:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x_glob) # I changed it to identity, real fc_norm is in the head
        else:
            x = self.norm(x) # This is before the cls token
            if self.return_features:
                return x
            outcome = x[:, 0]

        return outcome
    
    @torch.no_grad()
    def get_last_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        return x
    
    def make_dist(self, logits):
        logits = logits.reshape([-1, self.stoch, self.discrete])
        dist = td.Independent(td.OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        # Clumped down a lot, original mae uses old timm, latest timm is too new
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor = None) -> torch.Tensor:
        x = self.forward_features(x, delta_t)
        # Prior distribution only from current images p(z_{t+delta_t}|h_t,delta_t)
        prior_h = x[:, 0] # Use CLS tokens + time embedding 
        prior_logits = self.to_prior(prior_h) # Used in KL
        prior_dist = self.make_dist(prior_logits)
        prior_z = prior_dist.rsample() # Used in decoder
        prior_z = prior_z.reshape(*prior_z.shape[:-2], 1, self.stoch * self.discrete)
        
        for i in range(self.n_sample):
            temp_z = prior_dist.rsample()
            temp_z = temp_z.reshape(*temp_z.shape[:-2], 1, self.stoch * self.discrete)
            prior_z = torch.cat([prior_z, temp_z], dim=1)

        if self.n_sample > 1:
            prior_z = prior_z.mean(dim=1, keepdim=True)

        prior_z = self.decoder_embed_stoch(prior_z)
        x = self.decoder_embed(x)
        x = torch.cat([x, prior_z], dim=1)

        x = self.forward_head(x)
        return x



class ContrastiveViT3D(VisionTransformer3D):
    def __init__(self, use_learnable_pos_emb, **kwargs):
        super(ContrastiveViT3D, self).__init__(**kwargs)
        self.head =  torch.nn.Sequential(nn.Identity())
        self.initialize_weights()

        # TODO move these things to general vit3d
        self.patch_size = to_3tuple(kwargs['patch_size'])
        num_patches = self.patch_embed.num_patches
        self.num_prefix_tokens = 1 # There is always a cls token, but this is needed for registers
        if not use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, kwargs['embed_dim']), requires_grad=False)
            self.grid_shape = (kwargs['img_size'][0]//self.patch_size[0], kwargs['img_size'][1]//self.patch_size[1], kwargs['img_size'][2]//self.patch_size[2]) 
            pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_shape, cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    # MOCOv3 style
    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            trunc_normal_(m.weight, std=0.02)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return x


def vit3d_small_patch16( **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model = VisionTransformer3D(
        patch_size=(16,16,8), embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit3d_base_patch16(patch_size=(16,16,8),**kwargs):
    model = VisionTransformer3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit3d_large_patch16(patch_size=(16,16,8),**kwargs):
    model = VisionTransformer3D(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# TODO fix patch size
def vit3d_huge_patch14(**kwargs):
    model = VisionTransformer3D(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# Only difference is MOCOv3 stlye initialization
def contrvit3d_base_patch16(norm_layer = nn.LayerNorm, patch_size=(16,16,8), **kwargs):
    model = ContrastiveViT3D(use_learnable_pos_emb=False,
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(norm_layer, eps=1e-6), **kwargs)
    return model

def stoch_vit3d_base_patch16(patch_size=(16,16,8),**kwargs):
    model = stoch_VisionTransformer3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model