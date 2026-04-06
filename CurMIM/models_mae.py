# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from convGRU import ConvGRU
from vision_transformer import PatchEmbed, Block
from torch import Tensor
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.model_GRU = ConvGRU(input_size=(img_size/patch_size, img_size/patch_size),
                    input_dim=1,
                    hidden_dim=[64],
                    kernel_size=(3,3),
                    num_layers=1,
                    dtype=torch.cuda.FloatTensor,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)
        
        for param in self.model_GRU.parameters():
            param.requires_grad = True
            

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        decoder_blocks = [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)]
        self.decoder_blocks = nn.ModuleList(decoder_blocks)


        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.epoch_sim_epoch = []
        self.h_before_array = None
        self.h_i_array = None
        self.input_tensors = []
        self.mask_array = None
        self.grad_x = None
        self.GRU_map = None
        # --------------------------------------------------------------------------
        self.mask_ratios = []
        self.mask_MLP = nn.Sequential(
            nn.Linear(2 * (img_size//patch_size)**2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 1, bias=True)
        )
        self.mask_relu = nn.ReLU()
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
    
        self.initialize_weights()
        
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if(int(x.shape[1]**.5)**2 !=x.shape[1]):
            x = x[:,1:,:]
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, epoch, batch_num, init_mask, end_mask, total_epoch):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        mask_ratio = init_mask + (end_mask - init_mask) / total_epoch * epoch
        if epoch == 0:
            len_keep = int(L * (1 - mask_ratio))  
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # descend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            self.mask_array = mask
        else:
            height, width = self.input_tensors[batch_num].shape[-2], self.input_tensors[batch_num].shape[-1]
            hidden_dim = [64]
            if epoch == 1:
                len_keep = int(L * (1 - mask_ratio))  
                h = self.mask_array
                h_0 = torch.rand(h.shape[0], hidden_dim[0], height, width, device=x.device)
                init_states = []
                init_states.append(h_0)
            else:
                len_keep = int(L * (1 - mask_ratio))
                h_before = self.h_before_array
                h_i = self.h_i_array
                init_states = []
                init_states.append(h_before)
            input_tensor = self.input_tensors[batch_num].unsqueeze(1).detach()
            
            layer_output_list, last_state_list, h_m = self.model_GRU(input_tensor, init_states)
            
            layer_output_list = layer_output_list.requires_grad_()
            self.GRU_map = layer_output_list

            value, ids_shuffle = torch.sort(layer_output_list, dim=1, descending=False) 
            ids_restore = torch.argsort(ids_shuffle.detach(), dim=1)

            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            
            value = value + 5
            value[:, :len_keep] = -999
            mask = torch.gather(value, dim=1, index=ids_restore)
            
            self.h_before_array = last_state_list.detach()
            self.h_i_array = h_m.detach()
                
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, epoch, batch_num, init_mask, end_mask, total_epoch):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, epoch, batch_num, init_mask, end_mask, total_epoch)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore, epoch, batch_num):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        self.grad_x = x
        with torch.no_grad():
            cls_token = x[:,0,:]
            cls_token = cls_token.unsqueeze(1)
            patch_tokens = x[:,1:,:]
            cls_token_normalized = F.normalize(cls_token, p=2, dim=-1)
            patch_tokens_normalized = F.normalize(patch_tokens, p=2, dim=-1)
            sim = torch.matmul(cls_token_normalized, patch_tokens_normalized.transpose(1, 2))
            sim = sim.reshape(sim.shape[0],1,14,14)
            # sim:[64,196]
            if epoch == 0:
                self.epoch_sim_epoch.append(sim)
                self.input_tensors.append(sim)
            else:
                self.input_tensors[batch_num] = sim - self.epoch_sim_epoch[batch_num]
                self.epoch_sim_epoch[batch_num] = sim
            # print("input_tensor", self.input_tensors[batch_num])
            
        
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)

        
        # remove cls token
        # x = x[:, 1:, :]
        
        return x
    
    def forward_mask(self, epoch, loss, init_high, end_high, init_low, end_low, total_epoch, epsilon):
        N, L = loss.shape  # batch, length
        
        sigmma = epsilon - 0.01 * epoch
        top = int(L * (init_high + (end_high - init_high) / total_epoch * epoch))
        low = int(L * (init_low + (end_low - init_low) / total_epoch * epoch))
        
        high_value, _ = torch.sort(self.GRU_map, dim=1, descending=True) 
        high_sum = torch.sum(high_value[:, :top], dim=1).mean()
        low_sum = torch.sum(high_value[:, L - low:], dim=1).mean()
        mask_loss =  self.mask_relu(sigmma - (high_sum - low_sum))
        
        return mask_loss
            
    def gumbel_sigmoid(self, logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
        """
        Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
        The discretization converts the values greater than `threshold` to 1 and the rest to 0.
        The code is adapted from the official PyTorch implementation of gumbel_softmax:
        https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

        Args:
          logits: `[..., num_features]` unnormalized log probabilities
          tau: non-negative scalar temperature
          hard: if ``True``, the returned samples will be discretized,
                but will be differentiated as if it is the soft sample in autograd
         threshold: threshold for the discretization,
                    values greater than this will be set to 1 and the rest to 0

        Returns:
          Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
          If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
          be probability distributions.

        """
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0, 1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
        y_soft = gumbels.sigmoid()

        if hard:
            # Straight through.
            indices = (y_soft > threshold).nonzero(as_tuple=True)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
            y_hard[indices[0], indices[1]] = 1.0
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    
    def forward_loss(self, imgs, pred, mask, epoch, init_high, end_high, init_low, end_low, total_epoch, epsilon):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred[:,1:,:] - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss_mask = loss
        if epoch == 0:
            loss_all = (loss * mask).sum() / mask.sum()
        else:
            mask = self.gumbel_sigmoid(mask, hard = True)
            loss_support = self.forward_mask(epoch, loss_mask, init_high, end_high, init_low, end_low, total_epoch, epsilon)
            loss_all = (loss * mask).sum() / mask.sum() + 0.1 * loss_support
        return loss_all

    def forward(self, imgs, epoch, batch_num, init_mask, end_mask, total_epoch, init_high, end_high, init_low, end_low, epsilon):
        latent, mask, ids_restore = self.forward_encoder(imgs, epoch, batch_num, init_mask, end_mask, total_epoch)
        pred = self.forward_decoder(latent, ids_restore, epoch, batch_num)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, epoch, init_high, end_high, init_low, end_low, total_epoch, epsilon)
        return loss, pred, mask

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_deit_small_patch16_224(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_deit_small_patch16_224