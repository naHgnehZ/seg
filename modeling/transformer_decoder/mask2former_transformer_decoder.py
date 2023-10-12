# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, device = 'cpu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.device = device

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor], num_queries = 100, len_neg_query = 196, len_text_query = 100):
        if pos is None:
            return tensor
        else:
            if tensor.shape == pos.shape:
                return tensor + pos
            else:
                return tensor + torch.cat((pos, torch.zeros(len_neg_query+len_text_query, tensor.shape[1], tensor.shape[2]).to(device=self.device)), dim = 0)
 
    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     num_queries = 100,
                     len_neg_query = 196,
                     len_text_query = 100
                     ):
        
        q = k = self.with_pos_embed(tgt, query_pos, num_queries, len_neg_query, len_text_query)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    num_queries = 100,
                    len_neg_query = 196,
                    len_text_query = 100):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos, num_queries, len_neg_query, len_text_query)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                num_queries = 100,
                len_neg_query = 196,
                len_text_query = 100):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos, 
                                    num_queries, len_neg_query,
                                    len_text_query)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos,
                                 num_queries, len_neg_query,
                                    len_text_query)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, device = 'cpu'):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.device = device

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor], num_queries, len_neg_query, len_text_query):
        if pos is None:
            return tensor
        else:
            if tensor.shape == pos.shape:
                return tensor + pos
            else:
                return tensor + torch.cat((pos, torch.zeros(len_text_query+len_neg_query, tensor.shape[1], tensor.shape[2]).to(device=self.device)), dim = 0)

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     num_queries = 100,
                     len_neg_query = 196,
                     len_text_query = 100):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos, num_queries, len_neg_query, len_text_query),
                                   key=self.with_pos_embed(memory, pos, num_queries, len_neg_query, len_text_query),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    num_queries = 100,
                    len_neg_query = 196,
                    len_text_query = 100):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos, num_queries, len_neg_query, len_text_query),
                                   key=self.with_pos_embed(memory, pos, num_queries, len_neg_query, len_text_query),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                num_queries = 100,
                len_neg_query = 196,
                len_text_query = 100):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos,
                                    num_queries, len_neg_query,
                                    len_text_query)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos,
                                 num_queries, len_neg_query,
                                    len_text_query)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,  
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
        device='cpu'
        
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_classes = num_classes
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.device = device

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    device = self.device
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    device = self.device
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_classes = num_classes
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)


        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        ##### 修改默认值
        self.conv = nn.Conv2d(in_channels = 256, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mask_features, query_negative_features, query_text_input, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        neg_query = query_negative_features.permute(1, 0, 2)
        text_query = query_text_input.permute(1, 0, 2)
        len_neg_query = neg_query.shape[0]
        len_text_query = text_query.shape[0]

        predictions_class = []
        predictions_mask = []
        reconstruction = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(len_text_query, output, mask_features, attn_mask_target_size=size_list[0], need_res = False)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        cat_query = torch.cat((output, neg_query, text_query), dim = 0)
        
        # selfattn_mask = torch.zeros((396, 396)).to(device=self.device)
        tgt_size = self.num_queries + len_neg_query + len_text_query
        selfattn_mask = torch.ones(tgt_size, tgt_size).to(device=self.device) < 0
        # match query cannot see the reconstruct
        selfattn_mask[:self.num_queries, self.num_queries:] = True
        selfattn_mask[(self.num_queries + len_neg_query):, :(self.num_queries + len_neg_query)] = True
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: self-attention first
            output = self.transformer_self_attention_layers[i](
                cat_query,
                tgt_mask=selfattn_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
                num_queries = self.num_queries,
                len_neg_query = len_neg_query,
                len_text_query = len_text_query
            )

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], 
                query_pos=query_embed,
                num_queries = self.num_queries,
                len_neg_query = len_neg_query,
                len_text_query = len_text_query
            )

            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, outputs_res = self.forward_prediction_heads(len_text_query, output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], need_res = True)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            reconstruction.append(outputs_res)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'res_masks': reconstruction[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, reconstruction
            )
        }
        return out

    # output是learned query, mask_features是最后一个尺度特征, attn_mask_target_size是下一层所需的mask的尺度
    def forward_prediction_heads(self, len_text_query, output, mask_features, attn_mask_target_size, need_res):
        # output: QxNxC
        if need_res:
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            mask_output = decoder_output[:, :self.num_queries, :]
            res_output = decoder_output[:, self.num_queries : (self.num_queries + int((mask_features.shape[-1] / 4) ** 2)), :]
            outputs_class = self.class_embed(mask_output)
            mask_embed = self.mask_embed(mask_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            outputs_res = res_output.reshape(res_output.shape[0], int(mask_features.shape[-1] / 4), int(mask_features.shape[-1] / 4), res_output.shape[2])
            #conv = nn.Conv2d(in_channels = outputs_res.shape[3], out_channels=self.num_classes, kernel_size=1, stride=1, padding=0).to(device=self.device)
            outputs_res = self.conv(outputs_res.permute(0, 3, 1, 2))
            outputs_res = F.interpolate(outputs_res, size=(mask_features.shape[-1], mask_features.shape[-1]), mode="bilinear")
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            b, q, h, w = attn_mask.shape
            #attn_mask = torch.cat((torch.zeros((b, int((mask_features.shape[-1] / 4) ** 2) + self.num_queries, h, w)).to(device=self.device), attn_mask), 1)
            attn_mask = torch.cat((attn_mask, torch.zeros((b, int((mask_features.shape[-1] / 4) ** 2) + len_text_query, h, w)).to(device=self.device)), 1)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask, outputs_res

        else:
            decoder_output = self.decoder_norm(output)
            # NxQxC
            decoder_output = decoder_output.transpose(0, 1)
            outputs_class = self.class_embed(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            b, q, h, w = attn_mask.shape
            attn_mask = torch.cat((attn_mask, torch.zeros((b, int((mask_features.shape[-1] / 4) ** 2) + len_text_query, h, w)).to(device=self.device)), 1)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_res):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, 'res_masks': c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], [outputs_res[-1]]+outputs_res[:-1])
            ]
        else:
            return [{"pred_masks": b, 'res_masks': c} for b, c in zip(outputs_seg_masks[:-1],  [outputs_res[-1]]+outputs_res[:-1])]
