
from .projector import MLP
from .AttentionPool import AttentionPoolingClassifier
from .mae3d import mae3d_vit_small_patch16_d,mae3d_vit_small_patch16, mae3d_vit_base_patch16, mae3d_vit_base_patch16_ld, mae3d_vit_large_patch16
from .siammae3d import siam_mae3d_vit_small_patch16_d, siam_mae3d_vit_small_patch16, siam_mae3d_vit_base_patch16, siam_mae3d_vit_large_patch16,siam_mae3d_vit_base_patch16_ld, siam_mae3d_vit_base_patch16_md
from .vit3D import vit3d_base_patch16, vit3d_large_patch16, vit3d_small_patch16, vit3d_huge_patch14, stoch_vit3d_base_patch16
from .rsp3d import rsp_mae3d_vit_base_patch16, rsp_mae3d_vit_base_patch16_ld, rsp_mae3d_vit_base_patch16_md

### MICCAI 25 ablation
from .unconditional_ablation import uncond_ablation_mae3d_vit_base_patch16
from .encoder_conditional_siammae3d import ce_siam_mae3d_vit_base_patch16
from .dependent_conditional_siammae3d import dep_siam_mae3d_vit_base_patch16
from .independent_conditional_ablation import ind_cond_ablation_mae3d_vit_base_patch16
from .stoch_ee_dependent_conditional_ablation import stoch_ee_dep_cond_ablation_mae3d_vit_base_patch16
from .cls_encoder_conditional_siammae3d import cls_ce_siam_mae3d_vit_base_patch16
from .STAMP import stamp_vit_base

Models = {
    "AttnPoolHead": AttentionPoolingClassifier,
    "MLP": MLP,
    "mae3d_vit_small_patch16_d": mae3d_vit_small_patch16_d, # small decoder
    "mae3d_vit_small_patch16": mae3d_vit_small_patch16,
    "mae3d_vit_base_patch16": mae3d_vit_base_patch16,
    "mae3d_vit_base_patch16_ld": mae3d_vit_base_patch16_ld, # larger 3d decoder
    "mae3d_vit_large_patch16": mae3d_vit_large_patch16,
    "siam_mae3d_vit_small_patch16_d": siam_mae3d_vit_small_patch16_d, # small decoder
    "siam_mae3d_vit_small_patch16": siam_mae3d_vit_small_patch16,
    "siam_mae3d_vit_base_patch16": siam_mae3d_vit_base_patch16,
    "siam_mae3d_vit_base_patch16_md": siam_mae3d_vit_base_patch16_md, # medium decoder
    "siam_mae3d_vit_base_patch16_ld": siam_mae3d_vit_base_patch16_ld, # larger 3d decoder
    "siam_mae3d_vit_large_patch16": siam_mae3d_vit_large_patch16,
    "vit3d_small_patch16": vit3d_small_patch16,
    "vit3d_base_patch16": vit3d_base_patch16,
    "vit3d_large_patch16": vit3d_large_patch16,
    "vit3d_huge_patch14": vit3d_huge_patch14,
    "stoch_vit3d_base_patch16": stoch_vit3d_base_patch16,
    
    ### MICCAI 25 ablation
    "uncond_ablation_mae3d_vit_base_patch16": uncond_ablation_mae3d_vit_base_patch16,
    "ce_siam_mae3d_vit_base_patch16": ce_siam_mae3d_vit_base_patch16,
    "dep_siam_mae3d_vit_base_patch16": dep_siam_mae3d_vit_base_patch16,
    "ind_cond_ablation_mae3d_vit_base_patch16": ind_cond_ablation_mae3d_vit_base_patch16,
    "stoch_ee_dep_cond_ablation_mae3d_vit_base_patch16": stoch_ee_dep_cond_ablation_mae3d_vit_base_patch16,
    "cls_ce_siam_mae3d_vit_base_patch16": cls_ce_siam_mae3d_vit_base_patch16,
    'stamp_vit_base': stamp_vit_base,
    
    "rsp_mae3d_vit_base_patch16": rsp_mae3d_vit_base_patch16,
    "rsp_mae3d_vit_base_patch16_md": rsp_mae3d_vit_base_patch16_md, # medium decoder
    "rps_mae3d_vit_base_patch16_ld": rsp_mae3d_vit_base_patch16_ld, # larger 3d decoder
}