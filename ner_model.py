# 作者 ：duty
# 创建时间 ：2022/9/21 10:59 上午
# 文件 ：ner_model.py
import pickle
import torch
from transformers import BertTokenizer, BertPreTrainedModel
# from torchcrf import CRF
from torch.nn import Dropout, Linear, CrossEntropyLoss, Module, Embedding, LSTM
from transformers import BertModel
import torch

# from .layers.crf import CRF
from layers.crf import CRF
from layers.linears import PoolerStartLogits, PoolerEndLogits
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F


class SlotClassifier(Module):
    def __init__(self, input_dim, slot_class_cnt):
        super(SlotClassifier, self).__init__()
        self.dropout = Dropout(0.3)
        self.dense_layer = Linear(input_dim, slot_class_cnt)

    def forward(self, input):
        dropout_out = self.dropout(input)
        dense_out = self.dense_layer(dropout_out)
        return dense_out

## 正常的bert+fc
class NerModel(Module):
    def __init__(self, my_bert_model:BertModel, config, slot_class_cnt, use_crf,loss_type):
        super(NerModel, self).__init__()
        self.bert = my_bert_model.to(device)
        self.config = config
        self.input_dim = self.config.hidden_size ## bert的隐层神经元数
        self.slot_class_cnt = slot_class_cnt ## 实体类别数
        self.slot_net = SlotClassifier(self.input_dim, self.slot_class_cnt).to(device)
        self.crf = CRF(self.slot_class_cnt, batch_first=True).to(device)
        self.dropout = torch.nn.Dropout(0.3)
        self.loss_type = loss_type
        if self.loss_type == 'lsr':
            self.slot_loss_func = LabelSmoothingCrossEntropy(ignore_index=0)
        elif self.loss_type == 'focal':
            self.slot_loss_func = FocalLoss(ignore_index=0)
        else:
            self.slot_loss_func = CrossEntropyLoss(ignore_index=0)

        # self.slot_loss_func = CrossEntropyLoss()
        self.use_crf = use_crf

    ## 模型的输入和bert的输入一致
    def forward(self, input_ids, attention_masks, token_type_ids, slot_label_ids=None):
        bert_out = self.bert(input_ids, attention_masks, token_type_ids)
        sequence_out = bert_out[0]
        # pooler_out = bert_out[1]  ## bert的输出为 last_hidden_state:最后一层的输出， pooler_output:cls的输出 hidden_states:tuple of list，所有层的输出
        ## sequence_out用来做实体识别
        dropout_out = self.dropout(sequence_out)
        slot_out_logits = self.slot_net(dropout_out)
        # Only keep active parts of the loss
        slot_loss = 0
        if slot_label_ids is not None:
            if self.use_crf:
                slot_loss = self.crf(slot_out_logits, slot_label_ids, attention_masks, reduction='mean')
                slot_loss = -1*slot_loss
            else:
                if attention_masks is not None:
                    active_loss = attention_masks.view(-1) == 1
                    active_logits = slot_out_logits.view(-1, self.slot_class_cnt)[active_loss]
                    active_labels = slot_label_ids.view(-1)[active_loss]
                    slot_loss = self.slot_loss_func(active_logits, active_labels)
                else:
                    slot_loss = self.slot_loss_func(slot_out_logits.view(-1, self.slot_class_cnt), slot_label_ids.view(-1))
            # slot_loss = self.slot_loss_func(slot_out_logits.view(-1, self.slot_class_cnt), slot_label_ids.view(-1))
        total_loss = slot_loss
        return total_loss, slot_out_logits

## 参考https://lonepatient.top/2020/07/11/Boundary-Enhanced-Neural-Span-Classification-for-Nested-Named-Entity-Recognition.html
## https://github.com/lonePatient/BERT-NER-Pytorch
class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, my_bert_model:BertModel, config, num_labels, use_crf, loss_type):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = False
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.use_crf = use_crf
        self.bert = my_bert_model
        self.dropout = torch.nn.Dropout(0.3)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size+self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size+1 , self.num_labels)
        # self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        label_logits = F.softmax(start_logits, -1)
        if not self.soft_label:
            label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        return start_logits, end_logits


class LSTMForNer(torch.nn.Module):
    """
    LSTM for ner model
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, slot_class_cnt, use_crf, use_pretrained):
        super(LSTMForNer, self).__init__()
        self.embedding_dim = embedding_dim
        self.pretrained_words_embedding = pickle.load(open("pretrained_words_embedding", "rb"))
        self.pretrained_words_embedding = self.pretrained_words_embedding.astype(float)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.slot_class_cnt = slot_class_cnt
        self.use_crf = use_crf
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            self.embed_layer = Embedding(self.vocab_size, self.embedding_dim)
            self.embed_layer.weight.data.copy_(torch.from_numpy(self.pretrained_words_embedding))
            self.embed_layer.weight.requires_grad = True
        else:
            self.embed_layer = Embedding(self.vocab_size, self.embedding_dim)
        self.lstm_layer = LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dense_layer = Linear(self.hidden_dim*2, self.slot_class_cnt)
        self.crf = CRF(self.slot_class_cnt, batch_first=True)
        self.slot_loss_func = CrossEntropyLoss()

    def forward(self, input_ids, attention_masks, slot_label_ids=None):
        embed_out = self.embed_layer(input_ids)
        lstm_out, hiddens = self.lstm_layer(embed_out)
        slot_out_logits = self.dense_layer(lstm_out)
        slot_loss = 0
        if slot_label_ids is not None:
            if self.use_crf:
                slot_loss = self.crf(slot_out_logits, slot_label_ids, attention_masks, reduction='mean')
                slot_loss = -1 * slot_loss
            else:
                if attention_masks is not None:
                    active_loss = attention_masks.view(-1) == 1
                    active_logits = slot_out_logits.view(-1, self.slot_class_cnt)[active_loss]
                    active_labels = slot_label_ids.view(-1)[active_loss]
                    slot_loss = self.slot_loss_func(active_logits, active_labels)
                else:
                    slot_loss = self.slot_loss_func(slot_out_logits.view(-1, self.slot_class_cnt),
                                                    slot_label_ids.view(-1))

        return slot_loss, slot_out_logits


if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
