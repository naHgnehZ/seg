# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, get_world_size


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        res = False
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if res:
        inputs = F.softmax(inputs, dim=1) 
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks
    else:
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, device):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.device = device
        empty_weight = torch.ones(self.num_classes + 1).to(device)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)        

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, target_onehot, indices, num_masks, num_class):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(self.device)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, target_onehot, indices, num_masks, num_class):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"] # 
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )

        point_logits = src_masks[:, 0].flatten(1)
        point_labels = target_masks.flatten(1)       

        losses = {
            "loss_mask": sigmoid_focal_loss(point_logits, point_labels, num_masks), # sigmoid_ce_loss(point_logits, point_labels, num_masks), # 
            "loss_dice": dice_loss(point_logits, point_labels, num_masks)
        }

        del src_masks
        del target_masks
        return losses

    def res_loss_masks(self, outputs, targets, target_onehot, indices, num_masks, num_class):
        assert "res_masks" in outputs
        res_masks = outputs['res_masks']
        
        #target_onehot_o = target_onehot[:,1]
        
        target = torch.zeros(len(target_onehot), target_onehot[0].shape[0]-1, target_onehot[0].shape[1], target_onehot[0].shape[2]).to(res_masks.device)
        for i in range(len(target_onehot)):
            target[i] = target_onehot[i][1:]
        
        res_maskss = torch.zeros(target.shape).to(res_masks.device)
        
        for i in range(res_masks.shape[1]):
            res_mask = res_masks[:,i,:,:]
            res_mask = F.interpolate(
                res_mask[:,None], size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            res_mask = torch.squeeze(res_mask, dim=1)
            res_maskss[:,i,:,:] = res_mask
        
        point_logits = res_maskss.flatten(1)
        point_labels = target.flatten(1)
        
        losses = {
            "loss_res_mask": sigmoid_focal_loss(point_logits, point_labels, res_masks.shape[0]), # sigmoid_focal_loss(point_logits, point_labels, num_masks), # 
            "loss_res_dice": dice_loss(res_maskss, point_labels, res_masks.shape[0], res = True)
        }

        del res_masks
        del target
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x).long()
        target_onehot = target_onehot.scatter(dim=0, index = target.unsqueeze(0).long(), value=1)
        return target_onehot

    def get_loss(self, loss, outputs, targets, target_onehot, indices, num_masks, num_class):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'res_masks': self.res_loss_masks
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, target_onehot, indices, num_masks, num_class)

    def forward(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        targets, targets_onehot = self._get_targets(gt_masks)
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, targets_onehot, indices, num_masks, self.num_classes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, targets_onehot, indices, num_masks, self.num_classes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_targets(self, gt_masks):
        targets = []
        targets_onehot = []
        for mask in gt_masks:
            binary_masks_onehot = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = cls_label[:]
            binary_masks = binary_masks_onehot[labels]
            targets.append({'masks': binary_masks, 'labels': labels})
            targets_onehot.append(binary_masks_onehot)
        return targets, targets_onehot
        
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
