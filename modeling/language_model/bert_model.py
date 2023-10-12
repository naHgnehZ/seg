from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


## 做句子的分类 BertForSequence
class BertForSeq(BertPreTrainedModel):

    def __init__(self,config):  ##  config.json
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_feature = 256 # 类别数目
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, self.num_feature)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            #labels = None,
            return_dict = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ## loss损失 预测值preds

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )  ## 预测值

        pooled_output = outputs.last_hidden_state
        
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)

        return SequenceClassifierOutput(
            #loss=loss,  ##损失
            logits=logits,  ##softmax层的输入，可以理解为是个概率
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':

    ## 加载编码器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSeq.from_pretrained('bert-base-cased')
    ## 准备数据
    dev_dir = '/home/GuoY/language_model/train.txt'
    data = read_data(dev_dir)

    dev_dataset = InputDataSet(data, tokenizer=tokenizer, max_len=100)
    dev_dataloader = DataLoader(dev_dataset,batch_size=3,shuffle=False)
    ## 把数据做成batch
    batch = next(iter(dev_dataloader))
    ## 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 输入embedding
    #print(batch['input_ids'])
    #print(batch['input_ids'].shape)

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    #labels = batch['labels'].to(device)
    ## 预测
    model = model.to(device)
    model.eval()
    ## 得到输出
    outputs = model(input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict = True)
    ## 取输出里面的loss和logits
    logits = outputs.logits
    #loss = outputs.loss

    print(logits)
    print(logits.shape)
    #print(loss.item())

    #preds = torch.argmax(logits,dim=1)
    #print(preds)