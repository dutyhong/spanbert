# 作者 ：duty
# 创建时间 ：2021/5/10 7:38 下午
# 文件 ：bert_span_dataset.py
import pickle
# import sys,os

from my_bert_model import get_bert_model

# sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer



class SpanDataReader(object):
    def __init__(self,filepath, sample_cnt, tag="train"):
        ner_file = open(filepath, "r")
        self.sentences = []
        self.sent_slots = []
        self.tag2index = {}
        self.index2tag = {}

        ner_tag_map_file = open("./data/span_ner_tag_map", "r")
        for line in ner_tag_map_file.readlines():
            columns = line.split(",")
            self.tag2index[columns[0]] = int(columns[1])
            self.index2tag[int(columns[1])] = columns[0]
        ner_tag_map_file.close()


        for line in ner_file.readlines():
            columns = line.rstrip().split(";")
            # print(line.rstrip())
            # ori_sent = columns[1]
            stemmer_sent = columns[0]
            ner_result = columns[1]
            words = stemmer_sent.split(" ")
            ner_tag = ner_result.split(" ")
            sent_slot = []
            if len(words) != len(ner_tag):
                continue
            # cnt = cnt+1
            # if tag=="train" and cnt>sample_cnt:
            #     break
            # if tag=="test" and cnt<=sample_cnt:
            #     continue

            self.sentences.append(words)
            # self.sent_slots.append(ner_tag)
            for i, one_tag in enumerate(ner_tag):
                if one_tag=="O":
                    sent_slot.append(["O", 0, 0])
                else:
                    pos_name = one_tag.split("_")[0]
                    cate_name = one_tag.split("_")[1]
                    if pos_name=="S":
                        sent_slot.append([cate_name, self.tag2index[cate_name], self.tag2index[cate_name]])
                    if pos_name=="B":
                        sent_slot.append([cate_name, self.tag2index[cate_name], 0])
                    if pos_name=="E":
                        sent_slot.append([cate_name, 0, self.tag2index[cate_name]])
                    if pos_name=="I":
                        sent_slot.append([cate_name, 0, 0])
            self.sent_slots.append(sent_slot)

        self.tag_class_cnt = len(self.index2tag)
        pickle.dump(self.tag2index, open("./data/tag2index_dict.pkl", "wb"))
        pickle.dump(self.index2tag, open("./data/index2tag_dict.pkl", "wb"))


class BertSpanDataset(Dataset):
    def __init__(self, data_reader: SpanDataReader, max_seq_len: int,  tokenizer:BertTokenizer, file=None):
        # self.tokenizer = BertTokenizer.from_pretrained("/home/duty/publicdata/bert-base-uncased")
        self.tokenizer = tokenizer
        self.data_reader = data_reader
        self.max_seq_len = max_seq_len
        self.file = file

    def __getitem__(self, index):
        sentence = self.data_reader.sentences[index]
        sent_slots = self.data_reader.sent_slots[index]

        ## 做padding操作， 加入attention_masks, token_type_ids
        attention_masks = self.max_seq_len * [0]
        token_type_ids = self.max_seq_len * [0]

        ## start_ids, end_ids
        # sent_start_ids = [0] * self.max_seq_len
        # sent_end_ids = [0] * self.max_seq_len

        if len(sentence) > self.max_seq_len - 2:
            for i in range(self.max_seq_len):  ##最后的SEPtoken 忽略计算
                attention_masks[i] = 1
            sentence = [sentence[i] for i in range(self.max_seq_len - 2)]
            sent_start_ids = [sent_slots[i][1] for i in range(self.max_seq_len - 2)]
            # # sentence = ["CLS"]+sentence
            # # sentence = sentence+["SEP"]
            sent_start_ids = [0] + sent_start_ids
            sent_start_ids = sent_start_ids + [0]
            sent_end_ids = [sent_slots[i][2] for i in range(self.max_seq_len-2)]
            sent_end_ids = [0] + sent_end_ids
            sent_end_ids = sent_end_ids + [0]

        else:
            diff_len = self.max_seq_len - 2 - len(sentence)
            for i in range(len(sentence) + 2):
                attention_masks[i] = 1
            sent_start_ids = [sent_slots[i][1] for i in range(len(sentence))]
            sent_end_ids = [sent_slots[i][2] for i in range(len(sentence))]

            sent_start_ids = sent_start_ids+[0]*diff_len
            sent_end_ids = sent_end_ids+ [0]*diff_len
            sent_start_ids = [0] + sent_start_ids
            sent_start_ids = sent_start_ids + [0]

            sent_end_ids = [0] + sent_end_ids
            sent_end_ids = sent_end_ids + [0]
            sentence = sentence + diff_len * ["[PAD]"]
        bert_tokens = self.tokenizer.encode(sentence)
        return (torch.tensor(bert_tokens), torch.tensor(attention_masks), torch.tensor(token_type_ids),
                torch.tensor(sent_start_ids), torch.tensor(sent_end_ids))

    def __len__(self):
        return len(self.data_reader.sentences)


if __name__=="__main__":
    max_seq_len = 8
    config, tokenizer, bert_model = get_bert_model(False, "/home/tizi/publicdata/bert-base-uncased", max_seq_len)
    sample_cnt = 20000
    # data_reader = DataReader("../../data/save_train_data_v3.csv", sample_cnt)
    data_reader = SpanDataReader("../../data/train_all_file", sample_cnt, tag="train")
    # data_reader = DataReader("../../data/preprocessed_first_person_labeled_train_data", sample_cnt)
    dataset = BertSpanDataset(data_reader=data_reader, max_seq_len=max_seq_len, tokenizer=tokenizer)
    # file.close()
    dat_loader_iter = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    for batch_tokens, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids in dat_loader_iter:
        print(batch_tokens)
        # print(batch_start_ids)
        # print(batch_end_ids)
        print("dddd")
    print("dddd")