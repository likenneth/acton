import os
import math, bisect
from argparse import ArgumentParser
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

from plb.models.encoder import Transformer, Transformer_wote

dump_time = False
big_number = 2 ** 13  # a number >> T

class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    # from https://github.com/tk1980/LargeMarginInSoftmax/blob/master/models/modules/myloss.py
    def __init__(self, reg_lambda=0.3, deg_logit=None,
                 weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginInSoftmaxLoss, self).__init__(weight=weight, size_average=size_average,
                                                       ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0)  # number of samples
        C = input.size(1)  # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N), target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (C - 1)) * F.log_softmax(X, dim=1) * (1.0 - Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        # gather sizes on different GPU's
        size = torch.tensor(tensor.size(0), device=tensor.device)
        gathered_size = [torch.zeros_like(size) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_size, size)
        ctx.sizes = [_.item() for _ in gathered_size]
        max_bs = max(ctx.sizes)

        gathered_tensor = [tensor.new_zeros((max_bs, ) + tensor.shape[1:]) for _ in range(torch.distributed.get_world_size())]
        tbg = torch.cat([tensor, tensor.new_zeros((max_bs-tensor.size(0), ) + tensor.shape[1:])], dim=0)
        torch.distributed.all_gather(gathered_tensor, tbg)
        gathered_tensor = torch.cat([_[:s] for (_, s) in zip(gathered_tensor, ctx.sizes)], 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        my_rank = torch.distributed.get_rank()
        idx_from = sum(ctx.sizes[:my_rank])
        idx_to = idx_from + ctx.sizes[my_rank]
        return grad_input[idx_from:idx_to]


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class TAN(pl.LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            length: int,
            dataset: str,
            num_nodes: int = 1,
            arch: str = 'resnet50',
            hidden_mlp: int = 512,  # 2048, this is revised
            feat_dim: int = 128,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            temperature: float = 0.1,
            first_conv: bool = True,
            maxpool1: bool = True,
            optimizer: str = 'adam',
            lars_wrapper: bool = True,
            exclude_bn_bias: bool = False,
            start_lr: float = 0.,
            learning_rate: float = 1e-3,
            final_lr: float = 0.,
            weight_decay: float = 1e-6,
            val_configs=None,
            log_dir=None,
            protection=0,
            tr_layer=6,
            tr_dim=512,
            neg_dp=0.0,
            j=51, 
            **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size  # batch size from the view of scheduler
        self.real_batch_size = batch_size * length  # batch size from the view of optimizer

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr / 256 * self.real_batch_size
        self.final_lr = final_lr / 256 * self.real_batch_size
        self.learning_rate = learning_rate / 256  * self.real_batch_size
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.log_dir = log_dir
        # select which level of protection is used
        self.loss_calculator = [self.nt_xent_loss, self.nt_xent_loss_halfprotect, self.nt_xent_loss_protection, self.nt_xent_loss_rectangle, self.large_margin_loss_protection][protection]
        self.is_rectangle = protection == 3
        if protection == 4:
            self.lml = LargeMarginInSoftmaxLoss(reg_lambda=0.3)

        self.encoder = self.init_model(tr_layer=tr_layer, tr_dim=tr_dim, j=j)

        self.projection = Projection(input_dim=tr_dim, hidden_dim=tr_dim, output_dim=self.feat_dim)
        # originally using hidden_mlp for input_dim and hidden_dim
        self.dropout = nn.Dropout(p=neg_dp)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size * torch.cuda.device_count()
        if global_batch_size != 0:
            self.train_iters_per_epoch = math.ceil(self.num_samples / global_batch_size)
        else:
            self.train_iters_per_epoch = 0

        # define LR schedule
        warmup_lr_schedule = np.linspace(self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs)
        iters = np.arange(self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([
            self.final_lr + 0.5 * (self.learning_rate - self.final_lr) *
            (1 + math.cos(math.pi * t / (self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))))
            for t in iters
        ])

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        # construct validator
        self.validators = []
        if val_configs is not None and torch.cuda.device_count() != 0:
            for k, val in val_configs.items():
                val["log_dir"] = self.log_dir
                val["rank"] = self.global_rank
                val["world_size"] = torch.cuda.device_count()
                self.validators.append(construct_validator(k, val))

    def init_model(self, tr_layer, tr_dim, j):
        if self.arch == "Transformer":
            return Transformer(tr_layer, tr_dim, j)
        elif self.arch == "Tconv":
            # TODO: move to config
            config = get_default_tconv_net_config()
            config.tempconv_dim_in = 51
            config.tempconv_dim_out = 512
            config.tempconv_filter_widths = [5, ] * 5
            config.tempconv_channels = 1024
            return get_tconv_net(config)
        elif self.arch == "Transformer_wote":
            return Transformer_wote(tr_layer, tr_dim, j)
        else:
            assert 0, "Unknown model!"

    def forward(self, *args):
        x = self.encoder(*args)  # [N, T, f]
        if self.arch == "Tconv":
            x = x.permute(2, 0, 1).contiguous()
        return x

    def shared_step(self, batch):
        # img1, img2: [B, maxT1, 51], [B, maxT2, 51], maxT1 >= l1b, maxT2 >= l2b, any b in B
        # len1, len2: [B] of ints, real lengths, l1B, l2B
        # velo1, velo2: [B, maxT1], [B, maxT2], corresponding indices to video before temporal augmentation
        # m: [t1, t2]: real number between 0 and 1, t1 = sum_B l1b, t1 = sum_B l2b, composed of diagonal rectangles
        # chopped_bs: the batch size after reducing length difference to squares
        img1, img2, len1, len2, m, indices1, indices2, chopped_bs = batch
        # len1 and len2 actually the same
        h1_ = self(img1, len1)  # [maxT1, B, f=512]
        h2_ = self(img2, len2)
        h1_ = h1_.permute(1, 0, 2).contiguous()  # [B, maxT1, f=512]
        h2_ = h2_.permute(1, 0, 2).contiguous()

        if self.is_rectangle:
            dev = len1.device
            bs, maxT1, f = h1_.shape
            _, maxT2, _ = h2_.shape
            # big_boy = torch.arange(bs).to(dev).unsqueeze(-1) * big_number

            indices1 = torch.cat([torch.arange(l1b).to(dev) + b * maxT1 for b, l1b in enumerate(len1)], dim=0)
            h1 = torch.gather(h1_.flatten(0, 1), 0, indices1.unsqueeze(-1).repeat(1, f))  # [t1, f]
            # v1 = torch.gather((velo1 + big_boy).flatten(), 0, indices1)  # [t1]
            indices2 = torch.cat([torch.arange(l2b).to(dev) + b * maxT2 for b, l2b in enumerate(len2)], dim=0)
            h2 = torch.gather(h2_.flatten(0, 1), 0, indices2.unsqueeze(-1).repeat(1, f))  # [t2, f]
            # v2 = torch.gather((velo2 + big_boy).flatten(), 0, indices2)  # [t2]
            z1 = self.projection(h1)
            z2 = self.projection(h2)

            loss = self.loss_calculator(z1, z2, m, self.temperature)
        else:
            h1 = h1_.flatten(0, 1)[indices1]
            h2 = h2_.flatten(0, 1)[indices2]
            z1 = self.projection(h1)
            z2 = self.projection(h2)
            loss = self.loss_calculator(z1, z2, chopped_bs, self.temperature)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self, ):
        device = self.device
        self.eval()
        # self.cpu()
        for validator in self.validators:
            if self.global_rank != 0:
                save = -1
            else:
                if self.current_epoch % 500 == 474:  # note it has to be a subset of 5Z - 1, debug time use 0
                    save = self.current_epoch
                else:
                    save = -1

            metric_dict = validator(self, save=save)
            for name, metric in metric_dict.items():
                metric = torch.tensor([metric], device=device)
                self.log(name, metric, on_step=False, on_epoch=True, sync_dist=True)
        # self.to(device)
        if dump_time:
            if self.current_epoch % 1 == 0:
                torch.save(self.encoder, os.path.join(self.log_dir, f"dumped_at epoch{self.current_epoch}.ckpt"))

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'adam':
            # optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lars_wrapper:
            optimizer = LARSWrapper(
                optimizer,
                eta=0.001,  # trust coefficient
                clip=False
            )

        return optimizer

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]  # // torch.cuda.device_count()]

        # rank = torch.distributed.get_rank()
        # print(f"I am with rank {rank} and I am at global step {self.trainer.global_step}")

        # from lightning
        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer.to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)

    def nt_xent_loss(self, out_1, out_2, len, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        del len
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # Positive similarity, pos becomes [2 * batch_size]
        inner_sim = torch.exp(out_1 @ out_2.t().contiguous() / temperature)
        pos = torch.diagonal(inner_sim)
        pos = torch.cat([pos, pos], dim=0)

        # cov: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.exp(out @ out_dist.t().contiguous() / temperature)
        neg = cov.sum(dim=-1)  # length: \sum_i (l_1i + l_2i)

        # from each row, subtract e^(1/t) so that denominator has only t1 + t2 - 1 classes
        row_sub = torch.ones_like(neg) * math.exp(1 / temperature)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def nt_xent_loss_protection(self, out_1, out_2, len, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # Bg, total frame number on a certain GPU, = \sum bi, i over number of videos per GPU
        out = torch.cat([out_1, out_2], dim=0)  # [2Bg]
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)  # [2 \sum Bg]

        # Positive similarity, pos becomes [2 * batch_size]
        inner_sim = torch.exp(out_1 @ out_2.t().contiguous() / temperature)  # [Bg, Bg]
        pos = torch.diagonal(inner_sim)
        pos = torch.cat([pos, pos], dim=0)  # [2Bg]

        cov = torch.exp(out @ out_dist.t().contiguous() / temperature)  # [2Bg, 2 \sum Bg]
        neg = cov.sum(dim=-1)  # [2Bg]

        # from each row, subtract similarity to frame from the same video
        mask = torch.block_diag(*[pos.new_ones(_, _) for _ in len])
        mask = torch.cat([mask, mask], dim=0)
        mask = torch.cat([mask, mask], dim=1)
        outer_sim = torch.exp(out @ out.t().contiguous() / temperature)  # [2Bg, 2Bg]
        masked = outer_sim * mask
        row_sub = masked.sum(dim=0)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        loss = -torch.log(pos / (neg + pos + eps)).mean()

        return loss

    def large_margin_loss_protection(self, out_1, out_2, len, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # Bg, total frame number on a certain GPU, = \sum bi, i over number of videos per GPU
        out = torch.cat([out_1, out_2], dim=0)  # [2Bg]
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)  # [2 \sum Bg]

        # Positive similarity, pos becomes [2 * batch_size]
        inner_sim = torch.exp(out_1 @ out_2.t().contiguous() / temperature)  # [Bg, Bg]
        pos = torch.diagonal(inner_sim)
        pos = torch.cat([pos, pos], dim=0)  # [2Bg]

        cov = torch.exp(out @ out_dist.t().contiguous() / temperature)  # [2Bg, 2 \sum Bg]
        neg = cov.sum(dim=-1)  # [2Bg]

        # from each row, subtract similarity to frame from the same video
        mask = torch.block_diag(*[pos.new_ones(_, _) for _ in len])
        mask = torch.cat([mask, mask], dim=0)
        mask = torch.cat([mask, mask], dim=1)
        outer_sim = torch.exp(out @ out.t().contiguous() / temperature)  # [2Bg, 2Bg]
        masked = outer_sim * mask
        row_sub = masked.sum(dim=0)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        loss = -torch.log(pos / (neg + pos + eps)).mean()

        return loss

    def nt_xent_loss_halfprotect(self, out_1, out_2, len, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # Bg, total frame number on a certain GPU, = \sum bi, i over number of videos per GPU
        out = torch.cat([out_1, out_2], dim=0)  # [2Bg]
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)  # [2 \sum Bg]

        # Positive similarity, pos becomes [2 * batch_size]
        inner_sim = torch.exp(out_1 @ out_2.t().contiguous() / temperature)  # [Bg, Bg]
        pos = torch.diagonal(inner_sim)
        pos = torch.cat([pos, pos], dim=0)  # [2Bg]

        cov = torch.exp(out @ out_dist.t().contiguous() / temperature)  # [2Bg, 2 \sum Bg]
        neg = cov.sum(dim=-1)  # [2Bg]

        # from each row, subtract similarity to frame from the same video
        mask = torch.block_diag(*[pos.new_ones(_, _) for _ in len])
        mask_l = torch.cat([mask, mask.new_zeros(mask.shape)], dim=0)
        mask_r = torch.cat([mask.new_zeros(mask.shape), mask], dim=0)
        mask = torch.cat([mask_l, mask_r], dim=1)
        outer_sim = torch.exp(out @ out.t().contiguous() / temperature)  # [2Bg, 2Bg]
        masked = outer_sim * mask
        row_sub = masked.sum(dim=0)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def nt_xent_loss_rectangle(self, out_1, out_2, m, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [t1, f]
            out_2: [t2, f]
            m: [t1, t2]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        t1, t2 = m.shape
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1).detach()
            out_2_dist = SyncFunction.apply(out_2).detach()
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # Bg, total frame number on a certain GPU, = \sum bi, i over number of videos per GPU
        out = torch.cat([out_1, out_2], dim=0)  # [2Bg]
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)  # [2 \sum Bg]

        cov = torch.exp(torch.mm(out, out_dist.t().contiguous())/ temperature)  # [2Bg, 2 \sum Bg]
        cov = self.dropout(cov)
        neg = cov.sum(dim=-1)  # [2Bg]

        # # from each row, subtract e^(1/t) so that denominator has only t1 + t2 - 1 classes
        # row_sub = torch.ones_like(neg) * math.exp(1 / temperature)
        # neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # calculate positive
        inner_sim = torch.exp(torch.mm(out, out.t().contiguous())/ temperature)  # [2Bg, 2Bg]
        pos = (inner_sim * m).sum(dim=-1)

        loss = -torch.log(pos / (neg + eps)).mean()
        return loss
