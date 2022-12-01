# 作者 ：duty
# 创建时间 ：2021/5/11 10:46 上午
# 文件 ：bert_span_model_evaluate.py
import json

import pickle
import torch
from torch.utils.data import DataLoader

from data_process.bert_span_dataset import SpanDataReader, BertSpanDataset
from data_process.span_bert_output_decode import bert_extract_item
from my_bert_model import get_bert_model
from seqeval_evaluate import seq_eval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def decode_standard_tag(sent:str, entities:str):
    """
    将bert span的输出转换为标注seqeval 格式 gucci bag;S_brand S_category
    :param sent_words:
    :param entities:
    :return:
    """
    # print(sent)
    # print(entities)
    sent_words = sent.split(" ")
    sent_words.pop(0)
    sent_words.pop(-1)
    new_sent_words = []
    for sent_word in sent_words:
        if sent_word=="[PAD]":
            break
        new_sent_words.append(sent_word)
    tag_list = json.loads(entities)
    tag_sents = ["O"]*len(new_sent_words)
    for tag in tag_list:
        cate_name = tag[0]
        start_index = tag[1]
        end_index = tag[2]
        if end_index>=len(new_sent_words):
            continue
        if end_index-start_index==0:
            tag_sents[start_index] = "S_"+cate_name
        elif end_index-start_index==1:
            tag_sents[start_index] = "B_" + cate_name
            tag_sents[end_index] = "E_" + cate_name
        else :
            tag_sents[start_index] = "B_" + cate_name
            tag_sents[end_index] = "E_" + cate_name
            for i in range(end_index-start_index-1):
                tag_sents[start_index+i+1] = "I_"+cate_name
    return " ".join(new_sent_words), " ".join(tag_sents)

def evaluate_true_pred(sent, pred_tag, true_tag_dict):
    """
    关联测试集评估效果
    :param sent:
    :param tag_ent:
    :return:
    """
    if sent not in true_tag_dict:
        return 0, 0
    ori_tag = true_tag_dict[sent]
    ori_tag = ori_tag.split(" ")
    pred_tag = pred_tag.split(" ")
    acc_cnt = 0
    total_cnt = len(pred_tag)
    for ot, pt in zip(ori_tag, pred_tag):
        if ot==pt:
            acc_cnt = acc_cnt + 1
    return acc_cnt , total_cnt



# if __name__ == "__main__":
def model_eval(tag = "test"):
    max_seq_len = 12
    config, tokenizer, bert_model = get_bert_model(False, "./data/bert-base-uncased", max_seq_len)
    # config, tokenizer, bert_model = get_bert_model(False, "/home/tizi/publicdata/bert-medium", max_seq_len)
    sample_cnt = 0
    if tag=="test":
        file = open("./data/bert_span_evaluate_result", "w")
        data_reader = SpanDataReader("./data/test_samples", sample_cnt, tag="test")
        test_file = open("./data/test_samples", "r")
    else:
        file = open("./data/optimize_samples_evaluate_result", "w")
        data_reader = SpanDataReader("./data/new_optimize_samples", sample_cnt, tag="test")
        test_file = open("./data/new_optimize_samples", "r")

    # data_reader = DataReader("../../data/check_data/checked_data_updated_filter", sample_cnt)
    # test_file = open("../../data/check_data/checked_data_updated_filter", "r")
    true_tags_dict = {}
    cnt = 0
    for line in test_file.readlines():
        cnt = cnt +1
        if cnt<=sample_cnt:
            continue
        columns = line.rstrip().split(";")
        true_tags_dict[columns[0]] = columns[1]
    test_file.close()
    dataset = BertSpanDataset(data_reader=data_reader, max_seq_len=max_seq_len,  tokenizer=tokenizer)
    slot_class_cnt = len(data_reader.tag2index)
    data_loader_iter = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    nerBertSpanModel = torch.load("latest_bert_span_model.pt")
    nerBertSpanModel.eval()
    cnt = 0
    total_results = []

    acc_cnt = 0
    total_cnt = 0
    tmp_results = []
    for batch_tokens, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids in data_loader_iter:
        outputs = nerBertSpanModel(batch_tokens.to(device),
                        batch_attention_masks.to(device),batch_token_type_ids.to(device))
                        # batch_start_ids.to(device), batch_end_ids.to(device))
        start_logits, end_logits = outputs[:2]
        # batch_results = []
        for sent_tokens, start_logit, end_logit in zip(batch_tokens, start_logits, end_logits):
            R = bert_extract_item(start_logit, end_logit)
            if R:
                label_entities = [[data_reader.index2tag[x[0]], x[1], x[2]] for x in R]
            else:
                label_entities = []
            json_d = {}
            json_d['id'] = cnt
            cnt = cnt +1
            json_d['entities'] = label_entities
            total_results.append(json_d)
            # file.write(tokenizer.decode(sent_tokens)+";"+json.dumps(label_entities)+"\n")
            sent, tag_sent = decode_standard_tag(tokenizer.decode(sent_tokens),json.dumps(label_entities))
            sent_acc_cnt, sent_total_cnt = evaluate_true_pred(sent, tag_sent, true_tags_dict)
            acc_cnt = acc_cnt + sent_acc_cnt
            total_cnt = total_cnt + sent_total_cnt
            if sent_total_cnt!=0:
                file.write(sent+";"+true_tags_dict[sent]+";"+tag_sent+"\n")
                if tag=="increase":
                    tmp_results.append(sent+";"+true_tags_dict[sent]+";"+tag_sent)
    # 保存优化样本的结果，待下载
    if len(tmp_results)>0:
        pickle.dump(tmp_results,open("./data/optimize_samples_results", "wb"))
    file.close()
    print("准确率为：：%s"%(acc_cnt/total_cnt))
    # 计算实体结果准确率
    acc, precision, recall, f1 = seq_eval()
    return acc, precision, recall, f1