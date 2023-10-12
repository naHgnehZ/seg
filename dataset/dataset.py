#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2023/04/06 22:39:31
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import os
import json
import torch

import numpy as np
import random
from PIL import Image
from PIL import ImageOps

from copy import deepcopy

from .aug_strategy import imgaug_mask
from .aug_strategy import pipe_sequential_rotate
from .aug_strategy import pipe_sequential_translate
from .aug_strategy import pipe_sequential_scale
from .aug_strategy import pipe_someof_flip
from .aug_strategy import pipe_someof_blur
from .aug_strategy import pipe_sometimes_mpshear
from .aug_strategy import pipe_someone_contrast

# from .NuImages.nuimages import NuImages
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')
import sys
sys.setrecursionlimit(3000)


def read_textdata(file_path):
    question_file = file_path

    with open(question_file, 'r', encoding='utf-8') as f:
        questions = f.readlines()
        text = []

        for i in questions:
            if i == '\n':
                pass
            else:
                text.append(i[:-1])


    return text

def fill_paddings(data, maxlen):
    '''补全句长'''
    if len(data) < maxlen:
        pad_len = maxlen-len(data)
        paddings = [0 for _ in range(pad_len)]
        data = torch.tensor(data + paddings)
    else:
        data = torch.tensor(data[:maxlen])
    return data

class InputDataSet():

    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data[item])


        # 手动构建
        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        
        input_ids = fill_paddings(tokens_ids,self.max_len)
        attention_mask = [1 for _ in range(len(tokens_ids))]
        attention_mask = fill_paddings(attention_mask,self.max_len)
        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids,self.max_len)


        return {
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,

        }

        
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options        
        self.imgSizes = opt.INPUT.CROP.SIZE
        self.imgMaxSize = opt.INPUT.CROP.MAX_SIZE
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 2**5 # resnet 总共下采样5次

        # parse the input list
        if odgt is not None:
            self.parse_input_list(odgt, **kwargs)
        self.pixel_mean = np.array(opt.DATASETS.PIXEL_MEAN)
        self.pixel_std = np.array(opt.DATASETS.PIXEL_STD)

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.   
        img = (img - self.pixel_mean) / self.pixel_std
        img = img.transpose((2, 0, 1)) # [c, h, w]
        return img

    def segm_transform(self, segm: np.ndarray):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long()
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p
    
    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

    def resize_padding(self, img, outsize, Interpolation=Image.BILINEAR):
        w, h = img.size
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)
        ow, oh = round(w * ratio), round(h * ratio)
        img = img.resize((ow, oh), Interpolation)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
        return img

class Dataset(BaseDataset):
    def __init__(self, odgt, opt, dynamic_batchHW=False, whether_train=False, **kwargs):
        super(Dataset, self).__init__(odgt, opt, **kwargs)
        self.whether_train = whether_train
        self.root_dataset = opt.DATASETS.ROOT_DIR
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.MODEL.SEM_SEG_HEAD.COMMON_STRIDE # 网络输出相对于输入缩小的倍数
        self.dynamic_batchHW = dynamic_batchHW  # 是否动态调整batchHW, cswin_transformer需要使用固定image size
        self.num_querys = opt.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.max_len = opt.DATASETS.TEXT_MAX_LEN
        if whether_train:
            self.tokenizer = BertTokenizer.from_pretrained('pretrained_bert/bert-base-cased')
            # text_path = os.path.join(self.root_dataset, "dataset/COMOS/text/training/train.txt")
            text_path = "data/COMOS/text/training/train.txt"
            self.text = read_textdata(text_path)
            self.text_dataset = InputDataSet(self.text, tokenizer = self.tokenizer, max_len = self.max_len)
        # self.visualize = ADEVisualize()

        self.aug_pipe = self.get_data_aug_pipe()

    def get_data_aug_pipe(self):
        pipe_aug = []
        if random.random() > 0.5:
            aug_list = [pipe_sequential_rotate, pipe_sequential_scale, pipe_sequential_translate, pipe_someof_blur,
                        pipe_someof_flip, pipe_sometimes_mpshear, pipe_someone_contrast]
            index = np.random.choice(a=[0, 1, 2, 3, 4, 5, 6],
                                    p=[0.05, 0.25, 0.20, 0.25, 0.15, 0.05, 0.05])
            if (index == 0 or index == 4 or index == 5) and random.random() < 0.5:  # 会稍微削弱旋转 但是会极大增强其他泛化能力
                index2 = np.random.choice(a=[1, 2, 3], p=[0.4, 0.3, 0.3])
                pipe_aug = [aug_list[index], aug_list[index2]]
            else:
                pipe_aug = [aug_list[index]]
        return pipe_aug

    def get_batch_size(self, batch_records):
        batch_width, batch_height = self.imgMaxSize[0], self.imgMaxSize[1]

        if self.dynamic_batchHW:            
            if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
                this_short_size = np.random.choice(self.imgSizes)
            else:
                this_short_size = self.imgSizes

            batch_widths = np.zeros(len(batch_records), np.int32)
            batch_heights = np.zeros(len(batch_records), np.int32)
            for i, item in enumerate(batch_records):
                img_height, img_width = item['image'].shape[0], item['image'].shape[1]
                this_scale = min(
                    this_short_size / min(img_height, img_width), \
                    self.imgMaxSize[0] / max(img_height, img_width))
                batch_widths[i] = img_width * this_scale
                batch_heights[i] = img_height * this_scale
            
            batch_width = np.max(batch_widths)
            batch_height = np.max(batch_heights)
            
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        return batch_width, batch_height

    def __getitem__(self, index):        
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path).convert('L')
        
        # data augmentation            
        img = np.array(img)
        segm = np.array(segm)
        
        #for seq in self.aug_pipe:
            #img, segm = imgaug_mask(img, segm, seq)
    

        output = dict()
        output['image'] = img
        output['mask'] = segm
        
        if self.whether_train:
            neg_path = os.path.join(self.root_dataset, this_record['fpath_neg'])
            neg = Image.open(neg_path).convert('L')
            neg = np.array(neg)
            output['negative'] = neg
            output['text'] = self.text_dataset[index]['text']
            output['input_ids'] = self.text_dataset[index]['input_ids']
            output['attention_mask'] = self.text_dataset[index]['attention_mask']
            output['token_type_ids'] = self.text_dataset[index]['token_type_ids']


        return output

    def collate_fn(self, batch):
        batch_width, batch_height = self.get_batch_size(batch)
        if self.whether_train:
            out = {}
            images = []
            masks = []
            negatives = []
            input_idses = []
            attention_masks = []
            token_type_idses = []

            raw_images = []

            for item in batch:
                img = deepcopy(item['image'])
                segm = item['mask']
                neg = item['negative']

                img = Image.fromarray(img)
                segm = Image.fromarray(segm)
                neg = Image.fromarray(neg)

                #img = self.resize_padding(img, (batch_width, batch_height))
                img = self.img_transform(img)
                #segm = self.resize_padding(segm, (batch_width, batch_height), Image.NEAREST)
                
                # segm = segm.resize((item['image'].shape[0] // self.segm_downsampling_rate, item['image'].shape[0] // self.segm_downsampling_rate), Image.NEAREST)
                segm = segm
                #neg = self.resize_padding(neg, (batch_width, batch_height), Image.NEAREST)
                #neg = neg.resize((batch_width // self.segm_downsampling_rate, batch_height // self.segm_downsampling_rate), Image.NEAREST)

                images.append(torch.from_numpy(img).float())
                masks.append(torch.from_numpy(np.array(segm)).long())
                negatives.append(torch.from_numpy(np.array(neg)).long())
                input_idses.append(item['input_ids'])
                attention_masks.append(item['attention_mask'])
                token_type_idses.append(item['token_type_ids'])


                raw_images.append(item['image'])

            out['images'] = torch.stack(images)
            out['masks'] = torch.stack(masks)
            out['negatives'] = torch.stack(negatives)
            out['input_idses'] = torch.stack(input_idses)
            out['attention_masks'] = torch.stack(attention_masks)
            out['token_type_idses'] = torch.stack(token_type_idses)

            out['raw_img'] = raw_images
            return out        

        else:
            out = {}
            images = []
            masks = []

            raw_images = []

            for item in batch:
                img = deepcopy(item['image'])
                segm = item['mask']

                img = Image.fromarray(img)
                segm = Image.fromarray(segm)

                #img = self.resize_padding(img, (batch_width, batch_height))
                img = self.img_transform(img)
                #segm = self.resize_padding(segm, (batch_width, batch_height), Image.NEAREST)
                # segm = segm.resize((item['image'].shape[0] // self.segm_downsampling_rate, item['image'].shape[0] // self.segm_downsampling_rate), Image.NEAREST)
                segm = segm
                
                #neg = self.resize_padding(neg, (batch_width, batch_height), Image.NEAREST)
                #neg = neg.resize((batch_width // self.segm_downsampling_rate, batch_height // self.segm_downsampling_rate), Image.NEAREST)

                images.append(torch.from_numpy(img).float())
                masks.append(torch.from_numpy(np.array(segm)).long())

                raw_images.append(item['image'])

            out['images'] = torch.stack(images)
            out['masks'] = torch.stack(masks)

            out['raw_img'] = raw_images
            return out

    def __len__(self):
        return self.num_sample
    
