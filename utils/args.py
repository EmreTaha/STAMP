import argparse
import torch

def mae_args_parser():
    parser = argparse.ArgumentParser()    

    parser.add_argument('--data_dir', type=str, default="/optima/exchange/taha/harbor_center",
        help='data path to the volumes')

    parser.add_argument('--ssl_data_dir', type=str, default='/scratch/taha/ss_images_2_volume', #default='/home/optima/temre/preprocessing/Harbor/data/ss_images_2_volume',
        help='data path to the volumes')

    parser.add_argument('--fold', type=int, default=0,
        help='fold number to train on')

    parser.add_argument('--optim', type=str, default="AdamW",
        help='Optimizer type')

    parser.add_argument('--ag', default=False, action='store_true', required=False,
        help='whether use amsgrad with Adam or not')

    parser.add_argument('--exclude_nb', default=False, action='store_true', required=False,
        help='Exclude norm and biases')

    parser.add_argument('--save_dir', type=str, default="/home/optima/temre/preprocessing/Harbor/saved_models/volume/cnn_bilstm/experiment_28/",
        help='model and details save path')
    
    parser.add_argument('--epochs', type=int, default=100,
        help='max. num of epochs for training')
    
    parser.add_argument('--in_ch', type=int, default=1,
        help='number of input channels')

    parser.add_argument('--batch_size', type=int, default=16,
        help='batch size')

    parser.add_argument('--n_cl', type=int, default=1,
        help='number of classes, should be arranged accordingly with binning')

    parser.add_argument('--device',  default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        help="To use cuda, set to a specific GPU ID.")

    parser.add_argument('--lr', type=float, default=5e-5,
        help='Main learning rate')

    parser.add_argument("--lr_sch", type=str,
        help='What kind of learning rate scheduler to use')

    parser.add_argument('--grad_norm_clip', type=float, default=3.0,
        help='Enable gradient norm clipping')

    parser.add_argument('--backbone', type=str, default="mae_vit_base_patch16",
        help='Backbone model - currently vit 16 base are supported')

    parser.add_argument('--pretrained', default=False, action='store_true', required=False,
        help='Load pretrained backbone')

    parser.add_argument('--pretrained_model', type=str, default='',
        help='Path to the pretrained backbone weights')

    parser.add_argument('--cont', default=False, action='store_true', required=False,
        help='Shuffle the B-scan order. Only for ablation')

    parser.add_argument('--warmup_epochs', type=int, default=10,
        help='Number of warmup epochs')

    parser.add_argument('--wd', type=float, default=0,
        help='Weight decay')
    
    parser.add_argument('--ld', type=float, default=1.0,
        help='Layer decay, only for finetuning')

    parser.add_argument('--lw', type=float, default=5.0,
        help='Class weights of the loss')
    
    parser.add_argument('--num_workers', type=int, default=6,
        help='Number of workers for the dataloaders')

    parser.add_argument('--scale', default=False, action='store_true', required=False,
        help='Uses gradscaling and AMP')

    parser.add_argument('--lin_iters', type=int, default=20,
        help='max. num of epochs for linear classifier')

    parser.add_argument('--min_diff', type=int, default=90,
        help='Minimum number of difference between scans in days')

    parser.add_argument('--max_diff', type=int, default=540,
        help='Maximum number of difference between scans in days')
   
    parser.add_argument('--beta1', type=float, default=0.9,
        help='Beta1 of adam/lion')

    parser.add_argument('--beta2', type=float, default=0.95,
        help='Beta2 of adam')

    parser.add_argument('--mask_ratio', type=float, default=0.75,
        help='MAE masking ratio')  
    
    parser.add_argument('--perturb', type=float, default=0.0,
        help='Perturb future input in the SiamMAE setup')  
    
    parser.add_argument('--norm_pix_loss', default=False, action='store_true', required=False,
        help='Normalize pixel loss with the number of pixels')

    parser.add_argument('--global_pool', default=False, action='store_true', required=False,
        help='Global pooling in the backbone')
    
    parser.add_argument('--lin_bn', default=False, action='store_true', required=False,
        help='Attach a batch norm to the linear classifier')

    parser.add_argument('--sc', default='CS', type=str, required=False,#action='store_true', required=False,
        help='In cross attention block, Self attention first. This arguement will be deleted after testing')

    parser.add_argument('--sd', type=float, default=0.0,
        help='Stochastic depth probability')

    parser.add_argument('--rsp_mask', default=False, action='store_true', required=False,
        help='For RSP whether mask all future encoder inputs or not')
    
    parser.add_argument('--rsp_prior', default="straight", type=str, required=False,#action='store_true', required=False,
        help='Learnable prior distribution for RSP, two options: straight or soft')
    
    return parser