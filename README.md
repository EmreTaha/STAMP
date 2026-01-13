# Stochastic Siamese MAE Pretraining for Longitudinal Medical Images

## Requirements
- See the singularity .def file

## Dataset

***

## Training

### Self-supervised training
This script is specifically for STAMP, but we included model and loss definitions that are used in testing. The ablation studies and comparisons can be run in the similar manner, please see /src/models/... for all available models.

    python pretrain_STAMP.py --save_dir=/experiment_1 --epochs=800 --lr=0.0003 --lr_sch=cosine --wd=1e-2 --batch_size=48 --grad_norm_clip=3.0 --num_workers=12 --ssl_data_dir=/pretraining_data --exclude_nb  --backbone=STAMP --in_ch=1 --warmup_epochs=20 --min_diff=90 --max_diff=540 --beta2=0.95 --p_hflip=0.5
    
### AttentionPool Evaluation
After pretraining, you can run the evaluation. `--use_time_embed` enables TE during the inference. `--stoch_nsample` controls the number of stochastic sampling during the inference

    python attnpool_vit3d.py --save_dir=/attention_pool --data_dir=/supervised_data --fold=0 --pretrained --pretrained_model=/experiment_1/epoch_800.tar --epochs=200 --batch_size=128 --optim=AdamW --wd=0.0 --lw=10 --warmup_epochs=10 --lr=1e-3 --backbone=vit3d_base_patch16 --grad_norm_clip=3.0 --beta2=0.999 --num_workers=15 --use_stoch --stoch_nsample=1 --use_time_embed
    
## Citation

Please consider citing the paper if it is useful for you:

```
@article{emre2025stochastic,
  title={Stochastic Siamese MAE Pretraining for Longitudinal Medical Images},
  author={Emre, Taha and Chakravarty, Arunava and Pinetz, Thomas and Lachinov, Dmitrii and Menten, Martin J and Scholl, Hendrik and Sivaprasad, Sobha and Rueckert, Daniel and Lotery, Andrew and Sacu, Stefan and Schmidt-Erfurth, Ursula and Bogunović, Hrvoje},
  journal={arXiv preprint arXiv:2512.23441},
  year={2025}
}
```
