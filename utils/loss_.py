import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import os

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class EMDLoss(nn.modules.loss._Loss):
    def __init__(self,r=2):
        super(EMDLoss, self).__init__()
        self.r = r

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        cdf_y = torch.cumsum(y, dim=-1)
        cdf_pred = torch.cumsum(y_pred, dim=-1)

        cdf_diff = cdf_pred - cdf_y
        emd_loss = torch.mean(torch.pow(torch.abs(cdf_diff) + 1e-7, self.r),axis=-1)**(1. / self.r)
        return emd_loss.mean()

class VicLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, is_distributed: bool = False):
        super(VicLoss,self).__init__()
        self.lambd = λ
        self.mu = μ
        self.nu = ν
        self.gamma = γ
        self.eps = ϵ
        self.is_distributed = is_distributed

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        sim_loss = F.mse_loss(z1, z2)

        if self.is_distributed:
            z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
            z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)

        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # covariance loss
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / (D) #(2 * D)

        #diag = torch.eye(D, device=z1.device)
        #cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D 
        
        sim_lossd,std_lossd,cov_lossd =sim_loss.clone().detach(),std_loss.clone().detach(),cov_loss.clone().detach()

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_lossd, std_lossd, cov_lossd)

    def forward(self,z1,z2):
        return self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps)

class TemporalVicLoss(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1.):
        super(TemporalVicLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def timeloss(self,t1,t2,diff):
        return self.t*F.mse_loss(abs(t1-t2), diff)

    def forward(self,z1,z2,t1,t2,diff):
        if self.lambd or self.mu or self.nu:
            cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        else:
            cssl_loss, ind_loss = 0, (0,)
        time_loss = self.timeloss(t1,t2,diff)

        ind_loss = ind_loss + (time_loss,)
        return  cssl_loss+time_loss, ind_loss

class TemporalVicLossv2(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1.):
        super(TemporalVicLossv2,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def timeloss(self,t1,diff):
        return self.t*F.mse_loss(t1, diff)

    def forward(self,z1,z2,t1,diff):
        if self.lambd or self.mu or self.nu:
            cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        else:
            cssl_loss, ind_loss = 0, (0,)
        time_loss = self.timeloss(t1,diff)

        time_lossd = time_loss.clone().detach()
        ind_loss = ind_loss + (time_lossd,)
        loss = cssl_loss+time_loss
        return  loss, ind_loss
    
class EquimodLoss(VicLoss):
    # TODO add loss weights as hyperparameters
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super(EquimodLoss,self).__init__(λ, μ, ν, γ, ϵ)

    def forward(self,z1_inv,z2_inv,z2_equi,z_equi_pred):
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps) 
        equi_loss, ind_equi_loss = self.loss_fn(z2_equi, z_equi_pred, self.lambd, self.mu, self.nu, self.gamma, self.eps)

        total_loss = inv_loss + 0.45*equi_loss

        ind_loss = ind_inv_loss + ind_equi_loss
        return  total_loss, ind_loss
    
class EquimodLossv2(VicLoss):
    # Equivariance cov and var is calculated on z1_equi and z2_equi not on the predicted!!! Following https://github.com/facebookresearch/SIE
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super(EquimodLossv2,self).__init__(λ, μ, ν, γ, ϵ)

    def forward(self,z1_inv,z2_inv,z1_equi,z2_equi,z_equi_pred):
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps)
        equi_loss, ind_equi_loss = self.loss_fn(z1_equi, z2_equi, 0, self.mu, self.nu, self.gamma, self.eps)
        equi_sim_loss = F.mse_loss(z_equi_pred, z2_equi)
        equi_loss = equi_loss + self.lambd*equi_sim_loss

        total_loss = inv_loss + 0.45*equi_loss

        ind_loss = ind_inv_loss + (equi_sim_loss,) + ind_equi_loss[1:]
        return  total_loss, ind_loss
    
class TCLoss_abl(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1e-2):
        super(TCLoss_abl,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t

    def forward(self,z1_inv,z2_inv,r2, r_equi_pred):
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps) 
        equi_loss = F.mse_loss(r_equi_pred, r2)

        total_loss = inv_loss + self.lambd*self.t*equi_loss

        equi_lossd = equi_loss.clone().detach()
        ind_loss = ind_inv_loss + (equi_lossd,)

        return  total_loss, ind_loss

class TCLoss(VicLoss):
    #TODO indi thing is for softadapt delete after testing
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1e-2, tr: float = 1e-2, curve: float = 1.0, equiloss: str = 'mse'):
        super(TCLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t
        self.curve = curve
        self.tr = tr
        self.equiloss = equiloss

    def forward(self,z1_inv,z2_inv, r2, r_equi_pred, trajectory):
        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.lambd, self.mu, self.nu, self.gamma, self.eps)
        
        if self.equiloss == 'mse':
            equi_loss = F.mse_loss(r_equi_pred, r2)
        elif self.equiloss == 'symmse':
            equi_loss = 0.25*F.mse_loss(r_equi_pred.detach(), r2) + F.mse_loss(r_equi_pred, r2.detach())
        elif self.equiloss == 'cos':
            equi_loss = 1 - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        elif self.equiloss == 'ncos':
            equi_loss = - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        else:
            raise ValueError('Equiloss not recognized')
        
        norm = torch.norm(trajectory, dim=1)

        if self.tr != 0.0:
            traj_loss = torch.log1p((norm.neg() * self.curve).exp()).mean() # More numerically stable
        else:
            traj_loss = 0.0

        #total_loss = inv_loss + self.lambd*(self.t*equi_loss + self.tr*traj_loss)
        # NOTE After 03.07.2024 equiv loss weight is decoupled from lambd. DONT FORGET TO UPDATE args when running
        total_loss = inv_loss + self.t*equi_loss + self.tr*traj_loss

        equi_lossd, traj_lossd = equi_loss.clone().detach(), traj_loss.clone().detach()
        ind_loss = ind_inv_loss + (equi_lossd,) + (traj_lossd,)
        
        return  total_loss, ind_loss

import numpy as np
class TCLoss_guilherme(VicLoss):
    #TODO indi thing is for softadapt delete after testing
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1e-2, tr: float = 1e-2, curve: float = 1.0, equiloss: str = 'mse'):
        super(TCLoss_guilherme,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t
        self.curve = curve
        self.tr = tr
        self.equiloss = equiloss

    def forward(self,z1_inv,z2_inv, r2, r_equi_pred, trajectory):
        
        if self.equiloss == 'mse':
            equi_loss = F.mse_loss(r_equi_pred, r2)
        elif self.equiloss == 'symmse':
            equi_loss = 0.25*F.mse_loss(r_equi_pred.detach(), r2) + F.mse_loss(r_equi_pred, r2.detach())
        elif self.equiloss == 'cos':
            equi_loss = 1 - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        elif self.equiloss == 'ncos':
            equi_loss = - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        else:
            raise ValueError('Equiloss not recognized')
        
        #norm = torch.norm(trajectory, dim=1)
        #traj_loss = torch.log1p((norm.neg() * self.curve).exp()).mean() # More numerically stable

        #total_loss = inv_loss + self.lambd*(self.t*equi_loss + self.tr*traj_loss)
        total_loss = self.t*equi_loss
        
        return  total_loss, torch.from_numpy(np.array([0.0,0.0,0.0,equi_loss.cpu().clone().detach(),0.0],dtype=np.float32))
    
class TCLoss_schedule(VicLoss):
    #NOTE TC loss with lambda scheduler
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, t: float = 1e-2, tr: float = 1e-2, curve: float = 1.0, equiloss: str = 'mse',
                 warmup_epochs: int = 20):
        super(TCLoss_schedule,self).__init__(λ, μ, ν, γ, ϵ)
        self.t = t
        self.curve = curve
        self.tr = tr
        self.equiloss = equiloss
        self.eff_lambd = 0.0
        self.warmup_epochs = warmup_epochs
        self.curr_epoch = 0
        self.weight_increment = (self.lambd - self.eff_lambd) / self.warmup_epochs


    def forward(self,z1_inv,z2_inv, r2, r_equi_pred, trajectory, epoch):
        if self.curr_epoch < self.warmup_epochs:
            self.curr_epoch = epoch
            self.eff_lambd = self.weight_increment*self.curr_epoch
        else:
            self.eff_lambd = self.lambd

        inv_loss, ind_inv_loss = self.loss_fn(z1_inv, z2_inv, self.eff_lambd, self.mu, self.nu, self.gamma, self.eps)
        
        if self.equiloss == 'mse':
            equi_loss = F.mse_loss(r_equi_pred, r2)
        elif self.equiloss == 'cos':
            equi_loss = 1 - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        elif self.equiloss == 'ncos':
            equi_loss = - F.cosine_similarity(r_equi_pred, r2, dim=1).mean()
        
        norm = torch.norm(trajectory, dim=1)

        traj_loss = torch.log1p((norm.neg() * self.curve).exp()).mean() # More numerically stable

        #total_loss = inv_loss + self.lambd*(self.t*equi_loss + self.tr*traj_loss)
        # NOTE After 03.07.2024 equiv loss weight is decoupled from lambd
        total_loss = inv_loss + self.t*equi_loss + self.tr*traj_loss

        equi_lossd, traj_lossd = equi_loss.clone().detach(), traj_loss.clone().detach()
        ind_loss = ind_inv_loss + (equi_lossd,) + (traj_lossd,)
        
        return  total_loss, ind_loss



class ourLoss(VicLoss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, insensitive = "one"):
        super(ourLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.insensitive = insensitive

    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ, time_label):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        # original insensitive;
        # margin = F.mae_loss(z1, z2, reduction='none').mean(axis=1)-time_label
        margin = F.mse_loss(z1, z2, reduction='none').mean(axis=1)-time_label #mean is the across the representation dimension

        if self.insensitive == "two": margin = margin**2
        sim_loss = F.relu(margin).mean() # Reduction needs to be done because margin is specific to each example
        
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2

        z1 = z1 - z1.mean(dim=0) #Demeaning is added later!
        z2 = z2 - z2.mean(dim=0) #Demeaning is added later!

        # covariance loss
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / (D) #(2 * D)

        #diag = torch.eye(D, device=z1.device)
        #cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_tinz2[~diag.bool()].pow_(2).sum() / D 

        sim_lossd,std_lossd,cov_lossd =sim_loss.clone().detach(),std_loss.clone().detach(),cov_loss.clone().detach()

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_lossd, std_lossd, cov_lossd)
    
    def forward(self,z1,z2,time_label):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps, time_label) #calculate vicreg loss when we actually use them

        return  cssl_loss, ind_loss
    
class ourLossConc(VicLoss):
    # For the Variance loss both branches are concatanated
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, insensitive = "one"):
        super(ourLoss,self).__init__(λ, μ, ν, γ, ϵ)
        self.insensitive = insensitive

    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ, time_label):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        # original insensitive;
        # margin = F.mae_loss(z1, z2, reduction='none').mean(axis=1)-time_label
        margin = F.mse_loss(z1, z2, reduction='none').mean(axis=1)-time_label #mean is the across the representation dimension

        if self.insensitive == "two": margin = margin**2
        sim_loss = F.relu(margin).mean() # Reduction needs to be done because margin is specific to each example

        # variance loss
        # That is the difference compared to the original loss
        z_conc = torch.cat((z1,z2),dim=0)
        std_zconc = torch.sqrt(z_conc.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_zconc).mean()

        z1 = z1 - z1.mean(dim=0) #Demeaning is added later!
        z2 = z2 - z2.mean(dim=0) #Demeaning is added later!

        # covariance loss
        #z1 = z1 - z1.mean(dim=0)
        #z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / (D) #(2 * D)

        #diag = torch.eye(D, device=z1.device)
        #cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_tinz2[~diag.bool()].pow_(2).sum() / D 

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_loss,std_loss,cov_loss)

    def forward(self,z1,z2,time_label):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps, time_label) #calculate vicreg loss when we actually use them

        return  cssl_loss, ind_loss


class ourLossv2(VicLoss):
    # REDUCTED
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super(ourLossv2,self).__init__(λ, μ, ν, γ, ϵ)

    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ, time_label):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss
        margin = F.mse_loss(z1, z2, reduction='none').mean(axis=1)-time_label
        sim_loss = F.relu(margin).mean() + F.relu(-margin).mean()# Reduction needs to be done because margin is specific to each example
            
        # center features
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() / 2  + torch.relu(γ - std_z2).mean() / 2
            
        # covariance loss
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / (D) #(2 * D)

        #diag = torch.eye(D, device=z1.device)
        #cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D 

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_loss,std_loss,cov_loss)

    def forward(self,z1,z2,time_label):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps, time_label) #calculate vicreg loss when we actually use them

        return  cssl_loss, ind_loss

'''
class ContrTemporalVicLoss(VicLoss):
    # TODO not complete?
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4, margin = 0.4):
        super(TemporalVicLoss,self).__init__(λ, μ, ν, γ, ϵ)

    def loss_fn(self, z1, z2, time_label, λ, μ, ν, γ, ϵ):
        # Get batch size and dim of rep
        N,D = z1.shape
            
        # invariance loss (now it is a contrastive loss with time label)
        sim_loss = F.mse_loss(z1, z2)
        loss_contrastive = torch.mean((1-time_label) * torch.pow(sim_loss, 2) +
                                      (time_label) * torch.pow(torch.clamp(self.margin - sim_loss, min=0.0), 2))

        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() + torch.relu(γ - std_z2).mean()
            
        # covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N-1)
        cov_z2 = (z2.T @ z2) / (N-1)
        cov_loss = (self._off_diagonal(cov_z1).pow_(2).sum() + self._off_diagonal(cov_z2).pow_(2).sum()) / D

        #diag = torch.eye(D, device=z1.device)
        #cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D 

        return λ*loss_contrastive + μ*std_loss + ν*cov_loss, (loss_contrastive,std_loss,cov_loss)


    def forward(self,z1,z2,time_label):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, time_label, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        return  cssl_loss, ind_loss
'''
class NtXent(nn.modules.loss._Loss):
    def __init__(self,temperature, return_logits=False):
        super(NtXent, self).__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = F.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = F.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = F.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = F.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

class BarlowLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 0.0051):
        super(BarlowLoss,self).__init__()
        self.lambd = λ

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_fn(self, z1, z2):
        # cross-correlation matrix
        #c = (z1.T @ z2) / z1.shape[0]
        c = torch.matmul(z1.T, z2) / z1.shape[0]

        #TODO if multi gpu then add  torch.distributed.all_reduce(c) 

        on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
            
        # finall loss
        loss = on_diag + self.lambd * off_diag

        return loss, (on_diag,off_diag)

    def forward(self,z1,z2):
        return self.loss_fn(z1, z2)

class TemporalBarlowLoss(BarlowLoss):
    # TODO this is prototype
    def __init__(self, λ: float = 0.0051, t: float = 0.1):
        super(TemporalVicLoss,self).__init__(λ)
        self.t = t

    def timeloss(self,t1,diff):
        return self.t*F.mse_loss(t1, diff)

    def forward(self,z1,z2,t1,diff):
        cssl_loss, ind_loss = self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps) #calculate vicreg loss when we actually use them
        time_loss = self.timeloss(t1,diff)

        ind_loss = ind_loss + (time_loss,)
        return  cssl_loss+time_loss, ind_loss

class VibcLoss(nn.modules.loss._Loss):
    def __init__(self, λ: float = 25., μ: float = 25., ν: float = 10.,
                 γ: float = 1., ϵ: float = 1e-4):
        super(VibcLoss,self).__init__()
        self.lambd = λ
        self.mu = μ
        self.nu = ν
        self.gamma = γ
        self.eps = ϵ

    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    def covariance_loss(self, z1, z2):
        """Computes normalized covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """
        norm_z1 = z1 - z1.mean(dim=0)
        norm_z2 = z2 - z2.mean(dim=0)
        norm_z1 = F.normalize(norm_z1, p=2, dim=0)  # (batch * feature); l2-norm
        norm_z2 = F.normalize(norm_z2, p=2, dim=0)
        fxf_cov_z1 = torch.mm(norm_z1.T, norm_z1)  # (feature * feature)
        fxf_cov_z2 = torch.mm(norm_z2.T, norm_z2)
        fxf_cov_z1.fill_diagonal_(0.0)
        fxf_cov_z2.fill_diagonal_(0.0)
        cov_loss = (fxf_cov_z1 ** 2).mean() + (fxf_cov_z2 ** 2).mean()
        return cov_loss
    
    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ):
            
        # invariance loss
        sim_loss = F.mse_loss(z1, z2)
            
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = torch.relu(γ - std_z1).mean() + torch.relu(γ - std_z2).mean()
            
        # covariance loss
        cov_loss = self.covariance_loss(z1,z2)

        return λ*sim_loss + μ*std_loss + ν*cov_loss, (sim_loss,std_loss,cov_loss)

    def forward(self,z1,z2):
        return self.loss_fn(z1, z2, self.lambd, self.mu, self.nu, self.gamma, self.eps)

class ByolLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(ByolLoss,self).__init__()
    
    def loss_fn(self, z1, z2, p1, p2):
        loss1 = 2 - 2 * (z1 * p2).sum(dim=1)
        loss2 = 2 - 2 * (z2 * p1).sum(dim=1)
        return (loss1 + loss2).mean()
    
    def forward(self,z1,z2,p1,p2):
        return self.loss_fn(z1, z2, p1, p2)

class SimsiamLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(SimsiamLoss,self).__init__()
    
    def loss_fn(self, z1, z2, p1, p2):
        loss1 = - F.cosine_similarity(z1,p2, dim=1)
        loss2 = - F.cosine_similarity(z2,p1, dim=1)
        return (loss1.mean() + loss2.mean())*0.5
    
    def forward(self,z1,z2,p1,p2):
        return self.loss_fn(z1, z2, p1, p2)

#TODO implement this
'''    
class SimsiamTINCLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(SimsiamLoss,self).__init__()
    
    def loss_fn(self, z1, z2, p1, p2):
        loss1 = - F.cosine_similarity(z1,p2, dim=1)
        loss2 = - F.cosine_similarity(z2,p1, dim=1)
        return (loss1.mean() + loss2.mean())*0.5
    
    def forward(self,z1,z2,p1,p2):
        return self.loss_fn(z1, z2, p1, p2)
    
        margin = F.mse_loss(z1, z2, reduction='none').mean(axis=1)-time_label #mean is the across the representation dimension

        if self.insensitive == "two": margin = margin**2
        sim_loss = F.relu(margin).mean() 
'''
        

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass