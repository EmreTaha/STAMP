import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math
from functools import partial
from typing import Dict, List

def sgd_optimizer(model, lr, momentum, weight_decay, nesterov, exclude_bias_and_norm=False):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if 'bias' in key or 'bn' in key: # Dont apply wd to bias or bn
            if exclude_bias_and_norm: # Generally excluded in LARS
                continue
            apply_weight_decay = 0
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum, nesterov=nesterov)
    return optimizer

# Copied from Pytorch Lightning Bolts
# (https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/optimizers/lars.py)
class LARS(Optimizer):
    """Extends SGD in PyTorch with LARS scaling from the paper
    `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
        eps (float, optional): eps for division denominator (default: 1e-8)

    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> input = torch.Tensor(10)
        >>> target = torch.Tensor([1.])
        >>> loss_fn = lambda input, target: (input - target) ** 2
        >>> #
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        The application of momentum in the SGD part is modified according to
        the PyTorch standards. LARS scaling fits into the equation in the
        following fashion.

        .. math::
            \begin{aligned}
                g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
                v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \\end{aligned}

        where :math:`p`, :math:`g`, :math:`v`, :math:`\\mu` and :math:`\beta` denote the
        parameters, gradient, velocity, momentum, and weight decay respectively.
        The :math:`lars_lr` is defined by Eq. 6 in the paper.
        The Nesterov version is analogously modified.

    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded from
        layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
        and BYOL.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-8,
    ) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "trust_coefficient": trust_coefficient,
            "eps": eps,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                    lars_lr *= group["trust_coefficient"]

                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


def _channel_view(x) -> torch.Tensor:
    return x.reshape(x.size(0), -1)


def _layer_view(x) -> torch.Tensor:
    return x.reshape(1, -1)


def projection(p, grad, perturb, delta: float, wd_ratio: float, eps: float):
    wd = 1.
    expand_size = (-1,) + (1,) * (len(p.shape) - 1)
    for view_func in [_channel_view, _layer_view]:
        param_view = view_func(p)
        grad_view = view_func(grad)
        cosine_sim = F.cosine_similarity(grad_view, param_view, dim=1, eps=eps).abs_()

        # FIXME this is a problem for PyTorch XLA
        if cosine_sim.max() < delta / math.sqrt(param_view.size(1)):
            p_n = p / param_view.norm(p=2, dim=1).add_(eps).reshape(expand_size)
            perturb -= p_n * view_func(p_n * perturb).sum(dim=1).reshape(expand_size)
            wd = wd_ratio
            return perturb, wd

    return perturb, wd


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    '''
    Succesfully extorted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/adamp.py
    '''

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1.
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1. - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.add_(perturb, alpha=-step_size)

        return loss

def remove_bias_and_norm_from_weight_decay(parameter_groups: List[Dict]):
    # Pirated from somewhere I forgot

    out = []
    for group in parameter_groups:
        # parameters with weight decay
        decay_group = {k: v for k, v in group.items() if k != "params"}

        # parameters without weight decay
        no_decay_group = {k: v for k, v in group.items() if k != "params"}
        no_decay_group["weight_decay"] = 0.0
        group_name = group.get("name", None)
        if group_name:
            no_decay_group["name"] = group_name + "_no_decay"

        # split parameters into the two lists
        decay_params = []
        no_decay_params = []
        for param in group["params"]:
            if param.ndim <= 1:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # add groups back
        if decay_params:
            decay_group["params"] = decay_params
            out.append(decay_group)
        if no_decay_params:
            no_decay_group["params"] = no_decay_params
            out.append(no_decay_group)
    return out

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    # Sullied from timm
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class Lion(Optimizer):
    r"""Implements Lion algorithm."""
    # Extorted from timm

    def __init__(
            self,
            params,
            lr=1e-4,
            betas=(0.9, 0.99),
            weight_decay=0.0,
            maximize=False,
            foreach=None,
    ):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])

            lion(
                params_with_grad,
                grads,
                exp_avgs,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                maximize=group['maximize'],
                foreach=group['foreach'],
            )

        return loss

def lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        maximize: bool = False,
        foreach: bool = None,
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
):
    r"""Functional API that performs Lion algorithm computation.
    """
    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_lion
    else:
        func = _single_tensor_lion

    func(
        params,
        grads,
        exp_avgs,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
    )


def _single_tensor_lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            param = torch.view_as_real(param)

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Weight update
        update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
        param.add_(torch.sign(update), alpha=-lr)

        # Decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)


def _multi_tensor_lion(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
):
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

    # Perform stepweight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Weight update
    updates = torch._foreach_mul(exp_avgs, beta1)
    torch._foreach_add_(updates, grads, alpha=1 - beta1)

    updates = [u.sign() for u in updates]
    torch._foreach_add_(params, updates, alpha=-lr)

    # Decay the momentum running average coefficient
    torch._foreach_mul_(exp_avgs, beta2)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta2)



class MTAdam(Optimizer):
    r"""Implements MTAdam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MTAdam, self).__init__(params, defaults)

        self.total_grad = 0
        self.training_step = 0

    def __setstate__(self, state):
        super(MTAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    # assuming feature_map has requires_grad=True)
    # compared to adam, these are the added objects: loss_array, ranks, feature_map,
    @torch.no_grad()
    def step(self, loss_array, ranks, feature_map, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.update_weights(loss_array, ranks)

        return loss

    def update_weights(self, loss_array, ranks):

      for loss_index, loss in enumerate(loss_array):
        loss.backward(retain_graph=True)
        for group in self.param_groups:
          for p in group['params']:

            if p.grad is None:
              print("breaking")
              break

            if p.grad.is_sparse:
              raise RuntimeError('MTAdam does not support sparse gradients')

            amsgrad = group['amsgrad']

            state = self.state[p]

            # State initialization
            if len(state) == 0:
              state['step'] = 1
              for j, _ in enumerate(loss_array):
                # Exponential moving average of gradient values
                state['exp_avg'+str(j)] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'+str(j)] = torch.zeros_like(p.data)
                if amsgrad:
                  # Maintains max of all exp. moving avg. of sq. grad. values
                  state['max_exp_avg_sq'+str(j)] = torch.zeros_like(p.data)

                if j == 0: p.norms = [torch.ones(1).cuda()]
                else: p.norms.append(torch.ones(1).cuda())

            beta1, beta2, beta3 = group['betas']

            # normalize the norm of current loss gradients to be the same as the anchor
            if state['step'] == 1:
              p.norms[loss_index] = torch.norm(p.grad)
            else:
              p.norms[loss_index] = (p.norms[loss_index]*beta3) + ((1-beta3)*torch.norm(p.grad))
            if p.norms[loss_index] > 1e-10:
              for anchor_index in range(len(loss_array)):
                if p.norms[anchor_index] > 1e-10:
                  p.grad = ranks[loss_index] * p.norms[anchor_index] * p.grad / p.norms[loss_index]
                  break

            exp_avg, exp_avg_sq = state['exp_avg'+str(loss_index)], state['exp_avg_sq'+str(loss_index)]
            if amsgrad:
              max_exp_avg_sq = state['max_exp_avg_sq'+str(loss_index)]

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            if loss_index == len(loss_array) - 1:
              state['step'] += 1

            if group['weight_decay'] != 0:
              p.grad = p.grad.add(p, alpha=group['weight_decay'])

            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            if amsgrad:
              # Maintains the maximum of all 2nd moment running avg. till now
              torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
              # Use the max. for normalizing running avg. of gradient
              denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
              denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1

            if loss_index == 0 or not hasattr(p, 'exp_avg'):
              p.exp_avg = [exp_avg]
              p.denom = [denom]
              p.step_size = [step_size]
            else:
              p.exp_avg.append(exp_avg)
              p.denom.append(denom)
              p.step_size.append(step_size)
            if p.grad is not None:
              p.grad.detach_()
              p.grad.zero_()

      for group in self.param_groups:
        for p in group['params']:
          temp = 0
          max_denom = p.denom[0]
          for index in range(1, len(p.exp_avg)):
              max_denom = torch.max(max_denom, p.denom[index])

          for index in range(len(p.exp_avg)):
            update_step = -p.step_size[index]*(p.exp_avg[index]/max_denom)
            temp += update_step
          p.add_(temp)

      self.training_step += 1

OPTIMIZERS = {
    "SGD": partial(torch.optim.SGD, momentum=0.9, nesterov=True),
    "SGDexc": partial(sgd_optimizer, momentum=0.9, nesterov=True),
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RAdam": torch.optim.RAdam,
    "AdamP": AdamP,
    "MTAdam": MTAdam,
    "Lars": LARS,
    "Lion": Lion
}