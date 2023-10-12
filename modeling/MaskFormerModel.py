#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MaskFormerModel.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于DeformTransAtten的分割网络
'''

# here put the import lib
import torch
from torch import nn
from addict import Dict

from .backbone.resnet import ResNet, resnet_spec
from .backbone.swin import D2SwinTransformer
from .language_model.bert_model import BertForSeq

from .negative_feature_encoder.encoder import ViT_Encoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
# import argparse
# from fvcore.common.config import CfgNode
# from configs.config import Config
HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1

class MaskFormerHead(nn.Module):
    def __init__(self, cfg, input_shape, device):        
        super().__init__()        
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape)
        self.device = device
        self.predictor = self.predictor_init(cfg)
        
    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features =  cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                transformer_dropout,
                                                transformer_nheads,
                                                transformer_dim_feedforward,
                                                transformer_enc_layers,
                                                conv_dim,
                                                mask_dim,
                                                transformer_in_features,
                                                common_stride)
        return pixel_decoder

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        mask_classification = True
        predictor = MultiScaleMaskedTransformerDecoder( in_channels, 
                                                        num_classes, 
                                                        mask_classification,
                                                        hidden_dim,
                                                        num_queries,
                                                        nheads,
                                                        dim_feedforward,
                                                        dec_layers,
                                                        pre_norm,
                                                        mask_dim,
                                                        enforce_input_project,
                                                        self.device)
        return predictor

    def forward(self, features, query_negative_features, query_text_input, mask=None):

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)  
        # 进入transformer decoder层，需要mask_features(最后一个尺度特征), multi_scale_features(三个多尺度特征), mask(选择使用),
        # 相对于mask2former，需要额外引入negative Feature(额外的负样本特征), text Input(额外的文本提示)
        #query_negative_feat = torch.rand(100, 256)
        #query_text_input = torch.rand(100, 256)
        
        predictions = self.predictor(multi_scale_features, mask_features, query_negative_features, query_text_input, mask)        
        return predictions

class MaskFormerModel(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.backbone = self.build_backbone(cfg)
        self.ViT_Encoder = ViT_Encoder()
        self.language_model = BertForSeq.from_pretrained('pretrained_bert/bert-base-cased')
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape, device)

    def build_backbone(self, cfg):
        model_type = cfg.MODEL.BACKBONE.TYPE
        if model_type == 'resnet':            
            channels = [64, 128, 256, 512]
            if cfg.MODEL.RESNETS.DEPTH > 34:
                channels = [item * 4 for item in channels]
            backbone = ResNet(resnet_spec[model_type][0], resnet_spec[model_type][1])
            # backbone.init_weights()
            self.backbone_feature_shape = dict()
            for i, channel in enumerate(channels):
                self.backbone_feature_shape[f'res{i+2}'] = Dict({'channel': channel, 'stride': 2**(i+2)})
        elif model_type == 'swin':
            swin_depth = {'tiny': [2, 2, 6, 2], 'small': [2, 2, 18, 2], 'base': [2, 2, 18, 2], 'large': [2, 2, 18, 2]}
            swin_heads = {'tiny': [3, 6, 12, 24], 'small': [3, 6, 12, 24], 'base': [4, 8, 16, 32], 'large': [6, 12, 24, 48]}
            swin_dim = {'tiny':96, 'small': 96, 'base': 128, 'large': 192}
            cfg.MODEL.SWIN.DEPTHS = swin_depth[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.NUM_HEADS = swin_heads[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.EMBED_DIM = swin_dim[cfg.MODEL.SWIN.TYPE]
            backbone = D2SwinTransformer(cfg)
            self.backbone_feature_shape = backbone.output_shape()
        else:
            raise NotImplementedError('Do not support model type!')
        return backbone

    def forward(self, inputs, negative, text_input_ids, text_attention_mask, text_token_type_ids):
        
        image_features = self.backbone(inputs)
        negative_features = self.ViT_Encoder(inputs, negative)
        text_features = self.language_model(text_input_ids, text_attention_mask, text_token_type_ids, return_dict = True)
        
        outputs = self.sem_seg_head(image_features, negative_features, text_features.logits)
        return outputs


    
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='Mask2Former-mutimodel/configs/maskformer_nuimages.yaml')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument("--ngpus", default=1, type=int)
#     parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)

#     args = parser.parse_args()
#     cfg_ake150 = Config.fromfile(args.config)

#     cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True)    
#     cfg_base.update(cfg_ake150.__dict__.items())

#     cfg = cfg_base
#     for k, v in args.__dict__.items():
#         cfg[k] = v

#     cfg = Config(cfg)

#     cfg.ngpus = torch.cuda.device_count()
#     return cfg




# cfg = get_args()
# model = MaskFormerModel(cfg)
# image_data = torch.rand((3,3,224,224))
# negative = torch.rand((3,224,224))
# text_input_ids = torch.ones((3,100), dtype=torch.long)
# text_attention_mask = torch.ones((3,100), dtype=torch.long)
# text_token_type_ids = torch.ones((3,100), dtype=torch.long)
# outputs = model(image_data, negative, text_input_ids, text_attention_mask, text_token_type_ids)     






