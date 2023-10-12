#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   maskformer3D.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   DeformTransAtten分割网络训练代码
'''

# here put the import lib

import torch
import numpy as np
import os
import time
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch import distributed as dist
import sys
import itertools

from modeling.MaskFormerModel import MaskFormerModel
from utils.criterion import SetCriterion
from utils.matcher import HungarianMatcher
from utils.summary import create_summary
# from utils.solver import maybe_add_gradient_clipping
from utils.misc import load_parallal_model

class MaskFormer():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.device = torch.device("cuda", cfg.local_rank)
        self.is_training = cfg.MODEL.IS_TRAINING
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.last_lr = cfg.SOLVER.LR
        self.start_epoch = 0

        self.model = MaskFormerModel(cfg,device=self.device)
        
        
        if cfg.c and cfg.MODEL.PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.PRETRAINED_WEIGHTS):
            self.load_model(cfg.MODEL.PRETRAINED_WEIGHTS)
            print("loaded pretrain mode:{}".format(cfg.MODEL.PRETRAINED_WEIGHTS))

        self.model = self.model.to(self.device)
        # if cfg.ngpus > 1:
        #     self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)             

        self._training_init(cfg)


    def build_optimizer(self):
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim
            
        optimizer_type = self.cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                self.model.parameters(), self.last_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                self.model.parameters(), self.last_lr)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        # if not self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        #     optimizer = maybe_add_gradient_clipping(self.cfg, optimizer)

        return optimizer

    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights, map_location='cuda:0')
        print('loaded pretrained weights form %s !' % pretrain_weights)

        ckpt_dict = state_dict['model']
        self.last_lr = 6e-5 # state_dict['lr']
        self.start_epoch = 70 # state_dict['epoch']
        self.model = load_parallal_model(self.model, ckpt_dict)

    def _training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        res_mask_weight = cfg.MODEL.MASK_FORMER.RES_MASK_WEIGHT
        res_dice_weight = cfg.MODEL.MASK_FORMER.RES_DICE_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_res_mask": res_mask_weight, "loss_res_dice": res_dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "res_masks"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )

        self.summary_writer = create_summary(0, log_dir=cfg.TRAIN.LOG_DIR)
        self.save_folder = cfg.TRAIN.CKPT_DIR
        self.optim = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.9, patience=10)

    def reduce_mean(self, tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def train(self, train_sampler, data_loader, eval_loder, n_epochs):
        max_score = 0.88
        for epoch in range(self.start_epoch + 1, n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = self.train_epoch(data_loader, epoch)
            evaluator_score = self.evaluate(eval_loder)
            self.scheduler.step(evaluator_score)
            self.summary_writer.add_scalar('val_dice_score', evaluator_score, epoch)

            # if evaluator_score > max_score:
            if epoch%50==0:
                max_score = evaluator_score
                ckpt_path = os.path.join(self.save_folder, 'mask2former_Epoch{0}_dice{1:.4f}.pth'.format(epoch, max_score))
                save_state = {'model': self.model.state_dict(),
                              'lr': self.optim.param_groups[0]['lr'],
                              'epoch': epoch}
                torch.save(save_state, ckpt_path)
                print('weights {0} saved success!'.format(ckpt_path))
        self.summary_writer.close()

    def train_epoch(self,data_loader, epoch):
        self.model.train()
        self.criterion.train()
        load_t0 = time.time()
        losses_list = []
        loss_ce_list = []
        loss_dice_list = []
        loss_mask_list = []
        loss_res_dice_list = []
        loss_res_mask_list = []           
        
        for i, batch in enumerate(data_loader):                     
            inputs = batch['images'].to(device=self.device, non_blocking=True)
            targets = batch['masks']         
            negatives = batch['negatives'].to(device=self.device)
            text_input_ids = batch['input_idses'].to(device=self.device)
            text_attention_mask = batch['attention_masks'].to(device=self.device)
            text_token_type_ids = batch['token_type_idses'].to(device=self.device)

            outputs = self.model(inputs, negatives, text_input_ids, text_attention_mask, text_token_type_ids)                
            # targets[targets == 0] = 1
            # targets[targets == 255] = 2
            targets = targets + 1

            losses = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            
            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            loss_res_dice = 0.0
            loss_res_mask = 0.0
            for k in list(losses.keys()):
                if k in weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif 'ss_dice' in k:
                        loss_dice += losses[k]
                    elif 'ss_mask' in k:
                        loss_mask += losses[k]
                    elif 'res_dice' in k:
                        loss_res_dice += losses[k]
                    elif 'res_mask' in k:
                        loss_res_mask += losses[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            loss = loss_ce + loss_dice + loss_mask + loss_res_dice + loss_res_mask
            # loss = loss_ce + loss_dice + loss_mask
            with torch.no_grad():
                losses_list.append(loss.item())
                loss_ce_list.append(loss_ce.item())
                loss_dice_list.append(loss_dice.item())
                loss_mask_list.append(loss_mask.item())
                loss_res_dice_list.append(loss_res_dice.item())
                loss_res_mask_list.append(loss_res_mask.item())

            self.model.zero_grad()
            self.criterion.zero_grad()
            loss.backward()
            # loss = self.reduce_mean(loss, dist.get_world_size())
            self.optim.step()

            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optim.param_groups[0]['lr']
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.3f} loss_ce:{(np.mean(loss_ce_list)):.5f} loss_dice:{(np.mean(loss_dice_list)):.3f} loss_mask:{(np.mean(loss_mask_list)):.3f} loss_res_dice:{(np.mean(loss_res_dice_list)):.3f} loss_res_mask:{(np.mean(loss_res_mask_list)):.3f} , lr:{curent_lr:.2e} '
            
            
            # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f}, lr:{curent_lr:.2e}  '
            print(progress, end=' ')
            sys.stdout.flush()                
        
        self.summary_writer.add_scalar('loss', loss.item(), epoch)
        return loss.item()

    @torch.no_grad()                   
    def evaluate(self, eval_loder):
        self.model.eval()
        # self.criterion.eval()
        dice_score = []
        dice_score_2 = []
        
        for batch in eval_loder:
            inpurt_tensor = batch['images'].to(device=self.device, non_blocking=True)
            gt_mask = batch['masks']
            negatives = torch.zeros((batch['images'].shape[0], batch['images'].shape[-1], batch['images'].shape[-1])).to(device=self.device).long()
            text_input_ids = torch.zeros((batch['images'].shape[0], self.cfg.DATASETS.TEXT_MAX_LEN)).to(device=self.device).long()
            text_attention_mask = torch.zeros((batch['images'].shape[0], self.cfg.DATASETS.TEXT_MAX_LEN)).to(device=self.device).long()
            text_token_type_ids = torch.zeros((batch['images'].shape[0], self.cfg.DATASETS.TEXT_MAX_LEN)).to(device=self.device).long()

            outputs = self.model(inpurt_tensor, negatives, text_input_ids, text_attention_mask, text_token_type_ids)
            # gt_mask[gt_mask == 0] = 1
            # gt_mask[gt_mask == 255] = 2
            gt_mask = gt_mask + 1
            
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
                 
            mask_pred_resultss = torch.zeros(mask_pred_results.shape[0],mask_pred_results.shape[1],gt_mask.shape[-2],gt_mask.shape[-1]).to(self.device)
            for i in range(mask_pred_results.shape[1]):
                mask_pred_result = mask_pred_results[:,i,:,:]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:,None], size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)
                mask_pred_result = torch.squeeze(mask_pred_result, dim=1)
                mask_pred_resultss[:,i,:,:] = mask_pred_result
            
            
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_resultss)  
            
            int_pred_masks = torch.argmax(pred_masks.cpu(), axis=1) + 1
            int_pred_masks = self._get_binary_mask(int_pred_masks)
            gt_binary_mask = self._get_binary_mask(gt_mask)
               
            dice = self._get_dice(pred_masks.to(self.device), gt_binary_mask.to(self.device))
            dice_2 = self._get_dice(int_pred_masks.to(self.device), gt_binary_mask.to(self.device))
            dice_score.append(dice.item())
            dice_score_2.append(dice_2.item())
        score = np.mean(dice_score)
        score_2 = np.mean(dice_score_2)
        print('evaluate dice: {0}'.format(score))
        print('evaluate dice_2: {0}'.format(score_2))
        return score

    def _get_dice(self, predict, target):    
        smooth = 1e-5    
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = predict.sum(-1) + target.sum(-1) 
        score = (2 * num + smooth) / (den + smooth)
        return score.mean()

    def _get_binary_mask(self, target):
        # 返回每类的binary mask
        b, y, x = target.size()
        target_onehot = torch.zeros(b, self.num_classes + 1, y, x)
        target_onehot = target_onehot.scatter(dim=1, index=target.unsqueeze(1), value=1)
        
        return target_onehot[:,1:]

    def semantic_inference(self, mask_cls, mask_pred):       
        mask_cls = F.softmax(mask_cls, dim=-1)[...,1:]
        mask_pred = mask_pred.sigmoid() 
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        semseg = F.softmax(semseg, dim=1) 
        return semseg
