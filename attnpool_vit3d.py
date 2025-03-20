import pandas as pd
import numpy as np
import time
import os

from os import path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

import timm

from utils import create_cacheds_dl, Dataset_memm
from utils import Monai3DTransformsv3, bscan_sup_transformsv4, Barlow_augmentaions
from utils import save_last_k, load_last_k
from utils import pr_aucs,roc_aucs,f1_score, balanced_accs
from utils import convert_model
from sklearn.metrics import precision_recall_curve
from utils import initialize
from utils import mae_args_parser
from utils import interpolate_pos_embed, interpolate_2d_pos_embed_to_3d

from model import vit3d_base_patch16, vit3d_large_patch16, vit3d_huge_patch14, AttentionPoolingClassifier, stoch_vit3d_base_patch16

initialize()

parser = mae_args_parser()
parser.add_argument('--contrastive_pool', default=False, action='store_true', required=False,
    help='Makes sure that cls token is includedn in the pool')
parser.add_argument('--large_head_lr', default=False, action='store_true', required=False,
    help='Gives x10 more lr to the head')
parser.add_argument('--use_fixed_pos_emb', default=False, action='store_true', required=False,
    help='Whether to use fixed positional embeddings')
parser.add_argument('--use_time_embed', default=False, action='store_true', required=False,
    help='Whether to use time embeddings')
parser.add_argument('--best_metric', type=str, default="loss", required=False,
    help='best metric for early stopping')
parser.add_argument('--label', type=str, default="6m_label", required=False,
    help='label column')
parser.add_argument('--hg', default=False, action='store_true', required=False,
    help='whether use heavy augmentation or not')
parser.add_argument('--use_stoch', default=False, action='store_true', required=False,
    help='whether use stochasticity from pretraining or not')
parser.add_argument('--stoch_nsample', default=0, type=int, required=False,
    help='If stochasticity is used, how many samples to use')
parser.add_argument('--img_shape', default=448, type=int, required=False,
    help='If stochasticity is used, how many samples to use')
parser.add_argument('--adapt_ln', default=False, action='store_true', required=False,
    help='to adapt layner norms as well')
args = parser.parse_args()

if args.img_shape == 448:
    image_shape = (448,448)
    patch_size = (8,32,32)
else:
    image_shape = (224,224)
    patch_size = (8,16,16)

def prepare_data(args):

    args.save_dir = args.save_dir+'/fold_'+str(args.fold)

    os.makedirs(args.save_dir, exist_ok=True) 
    with open(args.save_dir+'/args.txt', 'w') as f:
        print(args.__dict__, file=f)

    # Start reading the dataset

    df_fov = pd.read_csv(args.data_dir+'/fovea.csv')

    # Create datasets and dataloaders
    df_train = pd.read_csv(args.data_dir+'/train_'+str(args.fold)+'.csv')
    df_train = df_train[df_train['Filepath'].str.contains(".dat")]

    df_test = pd.read_csv(args.data_dir+'/test_'+str(args.fold)+'.csv')
    df_test = df_test[df_test['Filepath'].str.contains(".dat")]

    df_val = pd.read_csv(args.data_dir+'/val_'+str(args.fold)+'.csv')
    df_val = df_val[df_val['Filepath'].str.contains(".dat")]

    train_paths = df_train['Filepath'].tolist()
    train_labels = df_train[args.label].tolist()
    train_scan = list(set(list(zip(train_paths,train_labels))))

    test_paths = df_test['Filepath'].tolist()
    test_labels = df_test[args.label].tolist()
    test_scan = list(set(list(zip(test_paths,test_labels))))

    val_paths = df_val['Filepath'].tolist()
    val_labels = df_val[args.label].tolist()
    val_scan = list(set(list(zip(val_paths,val_labels))))

    # End        
    NORM = [[0.1419264310816022], [0.09073713421821594]] # For unflattened
    if args.pretrained_model == "imagenet" or args.pretrained_model == "retfound":
        NORM = [[sum([0.485, 0.456, 0.406])/3.0], [sum([0.229, 0.224, 0.225])/3.0]] # For 2D pretrained models
    
    train_transform = Barlow_augmentaions(image_shape[0], scale=(0.4, 0.8), normalize=NORM, temporal=True, translation=False, p_blur=0, p_jitter=0.1, sol_threshold=0, p_horizontal_flip=0.5) if args.hg else bscan_sup_transformsv4(NORM,rotation=5,antialias=True)
    train_transform = Monai3DTransformsv3(train_transform,spatial_size=image_shape)
    
    test_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
                                   transforms.Normalize(*NORM)])
    test_transform = Monai3DTransformsv3(test_transform,spatial_size=image_shape)

    shape = (1024,512,128)
    train_scan_paths,train_scan_labels = map(list, zip(*train_scan))
    ds_train_scan = Dataset_memm(train_scan_paths,train_scan_labels,dif_fov=df_fov,shape=shape)

    val_scan_paths,val_scan_labels = map(list, zip(*val_scan))
    ds_val_scan = Dataset_memm(val_scan_paths,val_scan_labels,dif_fov=df_fov,shape=shape)

    test_scan_paths, test_scan_labels = map(list, zip(*test_scan))
    ds_test_scan = Dataset_memm(test_scan_paths,test_scan_labels,dif_fov=df_fov,shape=shape)

    trainloader = create_cacheds_dl(ds_train_scan, train_transform, cache_rate=1.0, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, drop_last=True, # persistent_workers=True,
                                         worker_fn = lambda id: np.random.seed(id + int(time.time())), progress=True)
    testloader = create_cacheds_dl(ds_test_scan, test_transform, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers-2,  cache_rate=1.0, progress=True)
    valloader = create_cacheds_dl(ds_val_scan, test_transform, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers-2,  cache_rate=1.0, progress=True)
    
    return trainloader,valloader,testloader

def train(args, trainloader, valloader, testloader):
    writer = SummaryWriter(args.save_dir)

    # Step 1: Create the lookup table
    inputs = np.arange(args.min_diff,args.max_diff+1, 30)
    normalized = (inputs - inputs.min()) / (inputs.max() - inputs.min())
    integer_values = np.arange(len(inputs))

    # Convert the lookup table to PyTorch tensors
    keys = torch.tensor(normalized, dtype=torch.float32)  # Normalized values as keys
    values = torch.tensor(integer_values, dtype=torch.float32)  # Integer values as values

    # Step 2: Define a function to find the closest key and map it to the value
    def map_to_lookup_table(batch: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Find the absolute differences between batch elements and keys
        diff = torch.abs(batch.unsqueeze(1) - keys)
        
        # Find the index of the closest key
        closest_idx = torch.argmin(diff, dim=1)
        
        # Map to corresponding values using the indices
        return values[closest_idx]

    # Handle save paths and initialize loggers
    if args.global_pool:
        Warning("Global pool is used, make sure to use the correct head!")

    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    
    # Define the model and load pretrained weights
    if "base" in args.backbone:
        model_f = vit3d_base_patch16
    elif "large" in args.backbone:
        model_f = vit3d_large_patch16
    elif "huge" in args.backbone:
        model_f = vit3d_huge_patch14
    # TODO move this to another file

    if args.use_stoch:
        model = stoch_vit3d_base_patch16(in_chans=args.in_ch, num_classes=args.n_cl, global_pool=args.global_pool, contrastive_cls=args.contrastive_pool, patch_size=patch_size, img_size=(32,)+image_shape,drop_path_rate=args.sd,
                    use_learnable_pos_emb=not args.use_fixed_pos_emb, time_embed=args.use_time_embed,return_features=True, n_sample = args.stoch_nsample)
    else:
        model = model_f(in_chans=args.in_ch, num_classes=args.n_cl, global_pool=args.global_pool, contrastive_cls=args.contrastive_pool, patch_size=patch_size, img_size=(32,)+image_shape,drop_path_rate=args.sd,
                        use_learnable_pos_emb=not args.use_fixed_pos_emb, time_embed=args.use_time_embed,return_features=True)

    if args.pretrained:
        # if it is imagenet or retfound, some projection to 3d is needed
        if args.pretrained_model == "imagenet" or args.pretrained_model == 'retfound':
            model = model_f(in_chans=args.in_ch, global_pool=args.global_pool, contrastive_cls=args.contrastive_pool, patch_size=patch_size, img_size=(32,)+image_shape,drop_path_rate=args.sd,
                        use_learnable_pos_emb=not args.use_fixed_pos_emb, time_embed=args.use_time_embed,return_features=True)
            if args.pretrained_model == "retfound":
                state = torch.load("RETFound_oct_weights.pth", map_location=torch.device('cpu'))['model']
            else:
                pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
                state = pretrained_model.state_dict()           
            if image_shape != (224,224): # Because imagenet and retfound is trained on 224x224 images
                del state['patch_embed.proj.weight'], state['patch_embed.proj.bias']
                args.ld = 1.0
            else:
                state['patch_embed.proj.weight'] = state['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1, patch_size[0], 1, 1) / patch_size[0]
                state['patch_embed.proj.weight'] = state['patch_embed.proj.weight'].mean(dim=1, keepdim=True)          
            state["pos_embed"] = interpolate_2d_pos_embed_to_3d(state["pos_embed"], num_patches_3d=(4,image_shape[-1]//patch_size[-1],image_shape[-1]//patch_size[-1]))
            msg = model.load_state_dict(state, strict=False)
            print("my message to you all:",msg)
            del state
            del pretrained_model
        else: # Else use saved weights
            saves = torch.load(args.pretrained_model, map_location=torch.device('cpu'))
            saves = saves['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in saves and saves[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del saves[k]
            interpolate_pos_embed(model, saves,orig_n_patch=(4,14,14)) #n_patch is the number of patch, not the size #TODO infer it from the given image and patch sizes...
            msg =model.load_state_dict(saves, strict=False)
    
    dim = 384 if args.use_stoch else 768
    model.head = AttentionPoolingClassifier(dim=dim,out_features=args.n_cl,linear_bias=True)
    
    if args.lin_bn:
        Warning("Batch norm is added to the linear layer not fine tuning!")
        #model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), 
        #                                        model.head)

    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    if torch.cuda.device_count() > 1:
        model = convert_model(model)
        model = nn.DataParallel(model)
        
    model.cuda()

    # Define optimizer and scheduler
    params = model.head.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)

    # Define Warmup scheduler
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it : (it+1)/((args.warmup_epochs+1)*len(trainloader)))
    
    # Define loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(args.lw, dtype=torch.float))

    # Run Training
    best_rauc = 0
    best_r6pr = 0
    best_valloss = 1000
    lr_hist = []
    train_losses, val_losses = [], []
    outputs= []
    best_prec,best_recall = [], []
    threshold = 0.5
    epoch_start = 0
    early_stop = 0

    if args.lr_sch is not None:
        print(args.lr_sch)
        if "step" in args.lr_sch:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.warmup_epochs)//5, gamma=0.2, last_epoch=-1)
        elif "cosine" in args.lr_sch:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warmup_epochs, eta_min=args.lr*1e-3)

    if args.cont:
        # Extract data
        saved_data = torch.load(path.join(args.save_dir, "last_epoch.tar"))
        model.load_state_dict(saved_data['model'], strict=True)
        optimizer.load_state_dict(saved_data['optim'])
        epoch_start = saved_data['epoch']
        if args.lr_sch is not None:
            if "step" in args.lr_sch:#TODO args["pretrained"]:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.warmup_epochs)//5, gamma=0.2, last_epoch=epoch_start-args.warmup_epochs)
            elif "cosine" in args.lr_sch:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epochs, last_epoch=epoch_start-args.warmup_epochs,
                                                        eta_min=args.lr*1e-3)
            lr_scheduler.load_state_dict(saved_data['sched'])
        outputs = saved_data['outputs']
        best_rauc = saved_data['best_rauc']

    # Mixed Precision
    use_amp = args.scale
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if not use_amp:
        #convert_model_to_fp32(model) #TODO fix
        pass
    
    # For Temporal Encoding prompting
    delta = 0.2 if args.label=='6m_label' else 0.6
    delta_t = torch.full((labels.size(0),), delta, dtype=torch.float32)
    delta_t = map_to_lookup_table(delta_t, keys, values)
    delta_t = delta_t.cuda()

    for epoch in range(epoch_start, args.epochs):

        # Run training
        model.head.train()
        running_loss = 0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        for i, inp_dict in enumerate(trainloader,0):
            
            inputs = inp_dict["image"]
            labels = inp_dict["label"]

            inputs, labels = inputs.cuda(), labels.cuda()
            if args.n_cl<2: labels = labels.reshape(-1,1)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outs = model.forward(inputs, delta_t)
                loss = criterion(outs, labels.float())
            scaler.scale(loss).backward() if args.scale else loss.backward()

            if bool(args.grad_norm_clip):
                if args.scale: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params,args.grad_norm_clip, error_if_nonfinite=True)
            
            # Update Optimizer
            scaler.step(optimizer) if args.scale else optimizer.step()
            if args.scale: scaler.update()

            # Warm-up update lr. Unlike cosine it is update at every "step"
            if epoch < args.warmup_epochs:
                scheduler_warmup.step()

            # Save loss and LR
            running_loss += loss.item()
            predicted = torch.sigmoid(outs).data.round()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        ep_time = time.time() - t0

        # Run validation
        val_loss = 0       
        correct = 0
        total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            model.eval()
            for j, inp_dict in enumerate(valloader,0):

                inputs = inp_dict["image"]
                labels = inp_dict["label"]

                delta_t = torch.full((labels.size(0),), 0.2, dtype=torch.float32)
                delta_t = map_to_lookup_table(delta_t, keys, values)
                delta_t = delta_t.cuda()

                inputs, labels = inputs.cuda(), labels.cuda()
                if args.n_cl<2: labels = labels.reshape(-1,1)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outs_vals = model.forward(inputs, delta_t)
                    batch_loss = criterion(outs_vals, labels.float())
                
                if args.scale: batch_loss = scaler.scale(batch_loss)
                val_loss += batch_loss.item()

                predicted = torch.sigmoid(outs_vals).data.round() 
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_labels += labels.reshape(-1).cpu()
                val_preds += torch.sigmoid(outs_vals).data.reshape(-1).cpu()

        train_losses.append(running_loss/(i+1))
        val_losses.append(val_loss/len(valloader))
        val_auc = roc_aucs(val_labels,val_preds)
        val_prauc = pr_aucs(val_labels, val_preds)
        val_f1 = f1_score(val_labels, np.around(val_preds))
        # Convert lists to numpy arrays
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # Define a range of possible thresholds
        thresholds = np.linspace(0, 1, num=100)

        # Find the best threshold
        val_baccs = 0
        best_threshold = 0 # for this specific epoch, naming is reversed
        for thresh in thresholds:
            # Binarize predictions based on the threshold
            binarized_predictions = (val_preds > thresh).astype(int)

            # Calculate accuracy
            accuracy = balanced_accs(val_labels, binarized_predictions)

            # Update best accuracy and threshold if current accuracy is higher
            if accuracy > val_baccs:
                val_baccs = accuracy
                best_threshold = thresh
        #Best threhold is found based on the validation set

        if (val_loss/len(valloader)) < best_valloss:
            best_valloss = val_loss/len(valloader)
            best_rauc = val_auc
            threshold = best_threshold
            save_last_k(model, epoch+1, args.save_dir, 1)
            best_prec,best_recall,_ = precision_recall_curve(val_labels, val_preds)
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > 20:
                print("Early stopping at epoch: ", epoch)
                break
        
        # Switch to Cosine Decay after warmup period and step every epoch
        if (epoch >= args.warmup_epochs) and (args.lr_sch is not None):
            lr_scheduler.step() 
            lr_hist.append(optimizer.param_groups[0]["lr"])

        # Log the epoch
        print(f"Epoch {epoch+1}/{args.epochs}.. "
                f"Train loss: {running_loss/len(trainloader):.3f}.. "
                f"Train accuracy: {train_correct/train_total:.3f}.. "
                f"Epoch time: {ep_time:.3f}.. "
                f"Val loss: {val_loss/len(valloader):.3f}.. "
                f"Val roc-auc: {val_auc:.3f}.. "
                f"Val balanced accuracy: {val_baccs:.3f}.. "
                f"Val pr-auc: {val_prauc:.3f}.. "
                f"Val F1: {val_f1:.3f}.. "
                f"Val accuracy: {correct / total:.3f}")

        outputs.append((f"Epoch {epoch+1}/{args.epochs}.. ",
                f"Train loss: {running_loss/len(trainloader):.3f}.. ",
                f"Train accuracy: {train_correct/train_total:.3f}.. ",
                f"Val loss: {val_loss/len(valloader):.3f}.. ",
                f"Val roc-auc: {val_auc:.3f}.. ",
                f"Val balanced accuracy: {val_baccs:.3f}.. ",
                f"Val pr-auc: {val_prauc:.3f}.. ",                    
                f"Val F1: {val_f1:.3f}.. ",
                f"Val accuracy: {correct / total:.3f}"))
        
        writer.add_scalar("Loss/train", running_loss/len(trainloader), epoch)
        writer.add_scalar("Loss/val", val_loss/len(valloader), epoch)
        writer.add_scalar("Accuracy/train", train_correct/train_total, epoch)
        writer.add_scalar("Accuracy/val", correct / total, epoch)
        writer.add_scalar("ROC-AUC/val", val_auc, epoch)
        writer.add_scalar("PR-AUC/val", val_prauc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("BalancedAccuracy/val", val_baccs, epoch)
            
        torch.save({'model': model.module.state_dict() if torch.cuda.device_count() > 1  else model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'sched': lr_scheduler.state_dict() if args.lr_sch is not None else "",
                        'outputs': outputs,
                        'epoch': epoch,
                        'best_rauc': best_rauc}, 
                    path.join(args.save_dir, "last_epoch.tar"))
    
    val_labels = [i.item() for i in val_labels]
    val_labels = np.array(val_labels)
    no_skill = len(val_labels[val_labels==1]) / len(val_labels)
    print("No skill pr-auc: ", no_skill)
    plt.clf() # TODO check if works
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(best_recall, best_prec, marker='.', label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(args.save_dir+'/pruac.png')
    
    # Test the best epoch
    if args.use_stoch:
        model = stoch_vit3d_base_patch16(in_chans=args.in_ch, num_classes=args.n_cl, global_pool=args.global_pool, contrastive_cls=args.contrastive_pool, patch_size=patch_size, img_size=(32,)+image_shape,drop_path_rate=args.sd,
                    use_learnable_pos_emb=not args.use_fixed_pos_emb, time_embed=args.use_time_embed,return_features=True, n_sample = args.stoch_nsample)
    else:
        model = model_f(in_chans=args.in_ch, num_classes=args.n_cl, global_pool=args.global_pool, contrastive_cls=args.contrastive_pool, patch_size=patch_size, img_size=(32,)+image_shape,drop_path_rate=args.sd,
                        use_learnable_pos_emb=not args.use_fixed_pos_emb, time_embed=args.use_time_embed,return_features=True)
        
    model.head = AttentionPoolingClassifier(dim=dim,out_features=args.n_cl,linear_bias=True)

    if torch.cuda.device_count() > 1:
        model = convert_model(model)
        model = nn.DataParallel(model)

    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    saves = load_last_k(args.save_dir,device)
    try:
        model.load_state_dict(saves, strict=True)
    except AttributeError:
        model.module.load_state_dict(saves, strict=True)
    
    test_loss = 0       
    correct = 0
    total = 0
    test_preds = []
    test_labels = []
    model.eval()
    with torch.no_grad():
        for j, inp_dict in enumerate(testloader,0):

            inputs = inp_dict["image"]
            labels = inp_dict["label"]

            inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels.reshape(-1,1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outs_test = model.forward(inputs, delta_t)
                batch_loss = criterion(outs_test, labels.float())
            
            if args.scale: batch_loss = scaler.scale(batch_loss)
            test_loss += batch_loss.item()

            predicted = torch.sigmoid(outs_test).data.round() 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if args.n_cl<2: labels = labels.reshape(-1)
            test_labels += labels.cpu()
            test_preds += torch.sigmoid(outs_test).data.reshape(-1).cpu()

    test_auc = roc_aucs(test_labels, test_preds)
    test_prauc =pr_aucs(test_labels, test_preds)
    test_f1 = f1_score(test_labels, np.around(test_preds))
    test_baccs = balanced_accs(test_labels, (test_preds> threshold).astype(int)) # use the validation threshold

    print(  f"Test roc-auc: {test_auc:.3f}.. "
            f"Test pr-auc: {test_prauc:.3f}.. "
            f"Test F1: {test_f1:.3f}.. "
            f"Test balanced accuracy: {test_baccs:.3f}.. "
            f"Test accuracy: {correct / total:.3f}")
        
    outputs.append((f"Test roc-auc: {test_auc:.3f}.. "
            f"Test pr-auc: {test_prauc:.3f}.. "
            f"Test balanced accuracy: {test_baccs:.3f}.. "
            f"Test F1: {test_f1:.3f}.. "
            f"Test accuracy: {correct / total:.3f}"))

    test_prec, test_recall, _ = precision_recall_curve(test_labels, test_preds)
    # plot the precision-recall curves
    test_labels = [i.item() for i in test_labels]
    test_labels = np.array(test_labels)
    no_skill = len(test_labels[test_labels==1]) / len(test_labels)
    print("No skill pr-auc: ", no_skill)
    plt.clf()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(test_recall, test_prec, marker='.', label='Model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(args.save_dir+'/test_pruac.png')

    writer.flush()

    f = open(args.save_dir+'/logs.txt', 'w')
    for t in outputs:
        line = ''.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()

    f = open(args.save_dir+'/preds.txt', 'w')
    test_preds = [(x.item(), y.item()) for x,y in zip(test_labels, test_preds)]
    for t in test_preds:
        line = ' '.join(str(c) for c in t)
        f.write(line + '\n')
    f.close()

def main():
    trainloader,valloader,testloader = prepare_data(args)
    train(args, trainloader, valloader, testloader)
    del trainloader,valloader,testloader

if __name__ == "__main__":
    main()