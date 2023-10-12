import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
import warnings
warnings.filterwarnings('ignore')
import sys
sys.setrecursionlimit(3000)


def read_data(file_path):
    question_file = file_path

    with open(question_file, 'r', encoding='utf-8') as f:
        questions = f.readlines()
        a = []
        b = []

        for i in questions:
            if i == '1\n' or i == '0\n':
                a.append(int(i[:-1]))
            elif i == '\n':
                pass
            else:
                b.append(i[:-1])

    return b

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
        
        # inputs = self.tokenizer.encode_plus(
        #     text,
        #     None,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     padding= 'max_length',
        #     truncation='longest_first',
        #     return_token_type_ids=True
        # )


        # return {
        #     'text':text,
        #     'input_ids':torch.tensor(inputs['input_ids'], dtype=torch.long),
        #     'attention_mask':torch.tensor(inputs['attention_mask'], dtype=torch.long),
        #     'token_type_ids':torch.tensor(inputs["token_type_ids"], dtype=torch.long)


        # }

        return {
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,

        }


if __name__ == '__main__':
    train_dir = '/home/GuoY/language_model/train.txt'
    dev_dir = '/home/GuoY/language_model/test.txt'
    model_dir = 'bert-base-cased'
    train = read_data(train_dir)
    test = read_data(dev_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_dataset = InputDataSet(train,tokenizer=tokenizer, max_len=128)
    train_dataloader = DataLoader(train_dataset,batch_size=4)
    batch = next(iter(train_dataloader))

    #print(batch['text'])
    #print(batch['input_ids'])
    #print(batch['attention_mask'].shape)
    #print(batch['token_type_ids'].shape)







