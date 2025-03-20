import time
import pandas as pd
import torch
import os
import numpy as np

from os import path

from torch.optim import lr_scheduler
from torch import nn
#from accelerate import Accelerator

from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader

from model import Models

from utils import Barlow_augmentaions, Monai3DTwoInpContrastTransformsv3
from utils import check_existing_model
from utils import mae_args_parser
from utils import Dataset_memm, TemporalDataset
from utils import convert_model
from utils import initialize

from timm.optim import create_optimizer_v2

initialize(allow_tf32=False)

parser = mae_args_parser()
parser.add_argument('--p_hflip', type=float, default=0.5,
    help='Random horizontal flip probability')
args = parser.parse_args()

def train(args):
    
    os.makedirs(args.save_dir, exist_ok=True) 

    optim_params = {'lr': args.lr,
                    'weight_decay': args.wd}#,
                    
    train_params = {'num_epochs': args.epochs, 'warmup_epchs': args.warmup_epochs, 'eta_min':args.lr*1e-4}

    writer = SummaryWriter(args.save_dir)

    with open(args.save_dir+'/args.txt', 'w') as f:
        print(args.__dict__, file=f)

    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

    
    image_shape = (448,448)
    patch_size = (8,32,32)

    model = Models[args.backbone](norm_pix_loss=args.norm_pix_loss,in_chans=args.in_ch, patch_size=patch_size, img_size=(32,)+image_shape, CS=args.sc, perturb=args.perturb).to(device) 
    #print(model)

    if torch.cuda.device_count() > 1:
        model = convert_model(model) #No.
        model = nn.DataParallel(model)
        
    model.cuda()

    # Create datasets and dataloaders
    NORM = [[0.1419264310816022], [0.09073713421821594]] # For unflattened

    # Pretraining transforms only crop and flip
    train_transf = Barlow_augmentaions(image_shape[0], temporal=True, scale=(0.9, 1.0), normalize=NORM, translation=False, # Temporal True means it returns a single barlow augs
                                       p_blur=0,p_solarize=0,p_jitter=0.1,p_horizontal_flip=args.p_hflip)

    train_transf = Monai3DTwoInpContrastTransformsv3(train_transf,shift_amount=5,shift_prob=0.5,spatial_size=image_shape) 

    df_fov = pd.read_csv(args.data_dir+'/fovea.csv')

    ssl_train_scan_paths = pd.read_csv(args.data_dir+'/ssl.csv')
    ssl_train_scan_paths = ssl_train_scan_paths['Filepath'].tolist()

    ssl_train_scan_paths = ssl_train_scan_paths
    ssl_ds = Dataset_memm(ssl_train_scan_paths, ssl_train_scan_paths, dif_fov=df_fov, shape=(1024,512,128)) # 3D shape
    ssl_temp_ds = TemporalDataset(ssl_ds, train_transf, cache_rate=1.0, min_max=(args.min_diff,args.max_diff),sorted=True)

    # Replacement true, to mimic repeated sampling to a degree
    sampler = torch.utils.data.RandomSampler(ssl_temp_ds, replacement=True, num_samples=len(ssl_temp_ds)*25) 
    ssl_dl = DataLoader(ssl_temp_ds, batch_size=args.batch_size, sampler=sampler, persistent_workers=False,
                                             worker_init_fn = lambda id: np.random.seed(id + int(time.time())), #shuffle=True,
                                           num_workers=args.num_workers, drop_last=True, pin_memory=False)

    # Define optimizer and scheduler
    optimizer = create_optimizer_v2(model,opt=args.optim.lower(), **optim_params, filter_bias_and_bn=args.exclude_nb, 
                                    amsgrad=args.ag, betas=(args.beta1,args.beta2))
    
    #warmup scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda it : (it+1)/(train_params['warmup_epchs']*len(ssl_dl)))

    # Check for existing training
    lp_acc = []
    loss_hist = []
    lr_hist = []

    epoch_start, saved_data = check_existing_model(args.save_dir, device)

    if saved_data:
        # Extract data
        msg = model.load_state_dict(saved_data['model'], strict=True)
        assert set(msg.missing_keys) == set()

        optimizer.load_state_dict(saved_data['optim'])

        if epoch_start >= train_params['warmup_epchs']:
            iters_left = iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_dl)
            curr_iter = (epoch_start-train_params['warmup_epchs'])*len(ssl_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, iters_left,
                                                        eta_min=train_params['eta_min'],
                                                        last_epoch=curr_iter)
        lp_acc = saved_data['lp_acc']
        loss_hist = saved_data['loss_hist']
        lr_hist = saved_data['lr_hist']

    # get total number of iterations
    total_iters = train_params['num_epochs'] * len(ssl_dl)
    print(len(ssl_dl))
    print(total_iters)
    # Run Training
    for epoch in range(epoch_start, train_params['num_epochs']):
        epoch_loss = 0
        model.train()
        start_time = time.time()

        for inp_dict in ssl_dl:
            optimizer.zero_grad()
    
            x1 = inp_dict["image"]
            x2 = inp_dict["image_1"]

            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass
            loss, _, _ = model(x1, x2, mask_ratio=args.mask_ratio)
            
            loss.backward()

            if bool(args.grad_norm_clip):
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_norm_clip, error_if_nonfinite=False)

            # Update Optimizer
            optimizer.step()
                       
            # Scheduler every iteration for cosine deday
            scheduler.step()

            # Save loss and LR
            epoch_loss += loss.item()
            lr_hist.extend(scheduler.get_last_lr())

        epoch_time = time.time() - start_time

        # Switch to Cosine Decay after warmup period
        if epoch+1==train_params['warmup_epchs']:
            iters_left = (train_params['num_epochs']-train_params['warmup_epchs'])*len(ssl_dl)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                        iters_left,
                                                        eta_min=train_params['eta_min'])
        
        # Log
        loss_hist.append(epoch_loss/len(ssl_dl))
        print(f'Epoch: {epoch}, Loss: {loss_hist[-1]}, Time epoch: {epoch_time}')
         
        writer.add_scalar("Loss", loss_hist[-1], epoch)
        writer.add_scalar("LR/scheduler", lr_hist[-1], epoch)

        # save
        if (epoch+1)%(100)==0 or (epoch+1)==train_params['num_epochs']:
            
            torch.save({'model':model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': scheduler.state_dict(),
                        'lp_acc': lp_acc,
                        'loss_hist': loss_hist,
                        'lr_hist': lr_hist}, 
                    path.join(args.save_dir, f'epoch_{epoch+1:03}.tar'))

    # Saving at the last epoch
    
    torch.save({'model':model.state_dict(),
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict(),
                'lp_acc': lp_acc,
                'loss_hist': loss_hist,
                'lr_hist': lr_hist}, 
            path.join(args.save_dir, f'epoch_{epoch+1:03}.tar'))
    writer.flush()

def main():
    train(args)

if __name__ == "__main__":
    main()