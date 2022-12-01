# 作者 ：duty
# 创建时间 ：2022/9/15 4:52 下午
# 文件 ：load_model.py
# 作者 ：duty
# 创建时间 ：2021/5/10 9:03 下午
# 文件 ：bert_span_model_train.py
import json

from torch.utils.data import DataLoader

from data_process.bert_span_dataset import SpanDataReader, BertSpanDataset
from losses.label_smoothing import LabelSmoothingCrossEntropy
from my_bert_model import get_bert_model
import torch

from ner_model import BertSpanForNer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ner_model_train(model, data_iter, iter_epochs, slot_class_cnt):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.to(device)
    model.train()
    for i in range(iter_epochs):
        print("第%s个epoch结果如下：：" % (i + 1))
        num = 0
        for batch_tokens, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids in data_iter:
            num = num + 1
            optimizer.zero_grad()
            outputs = model(batch_tokens.to(device),
                            batch_attention_masks.to(device),batch_token_type_ids.to(device))
            start_logits, end_logits = outputs[0], outputs[1]

            loss_fct = LabelSmoothingCrossEntropy()
            tmp_start_logits = start_logits.view(-1, slot_class_cnt)
            tmp_end_logits = end_logits.view(-1, slot_class_cnt)
            active_loss = batch_attention_masks.to(device).view(-1) == 1
            active_start_logits = tmp_start_logits[active_loss]
            active_end_logits = tmp_end_logits[active_loss]

            active_start_labels = batch_start_ids.to(device).view(-1)[active_loss]
            active_end_labels = batch_end_ids.to(device).view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            print(total_loss.item())
            total_loss.backward()
            optimizer.step()
    torch.save(model, "latest_bert_span_model.pt")
    torch.save(model.state_dict(), "latest_bert_span_model.params")


# if __name__ == "__main__":
def train_main():
    max_seq_len = 12
    epochs = 30
    config, tokenizer, bert_model = get_bert_model(False, "./data/bert-base-uncased", max_seq_len)
    data_reader = SpanDataReader("./data/latest_version_samples", sample_cnt=10000000, tag="train")
    dataset = BertSpanDataset(data_reader=data_reader, max_seq_len=max_seq_len, tokenizer=tokenizer)
    # file.close()
    slot_class_cnt = len(data_reader.tag2index)
    data_loader_iter = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    nerBertSpanModel = BertSpanForNer(bert_model, config, slot_class_cnt, use_crf=True, loss_type="lsr")
    ner_model_train(nerBertSpanModel, data_loader_iter, epochs,  slot_class_cnt)
    # nerBertSpanModel = torch.load("bert_span_model.pt")
    # nerBertSpanModel.eval()
    # cnt = 0
    # total_results = []
    # file = open("../../data/bert_span_evaluate_result", "w")
    # for batch_tokens, batch_attention_masks, batch_token_type_ids, batch_start_ids, batch_end_ids in data_loader_iter:
    #     outputs = nerBertSpanModel(batch_tokens.to(device), batch_token_type_ids.to(device),
    #                     batch_attention_masks.to(device),
    #                     batch_start_ids.to(device), batch_end_ids.to(device))
    #     total_loss, start_logits, end_logits = outputs[:3]
    #     # batch_results = []
    #     for sent_tokens, start_logit, end_logit in zip(batch_tokens, start_logits, end_logits):
    #         R = bert_extract_item(start_logit, end_logit)
    #         if R:
    #             label_entities = [[data_reader.index2tag[x[0]], x[1], x[2]] for x in R]
    #         else:
    #             label_entities = []
    #         json_d = {}
    #         json_d['id'] = cnt
    #         cnt = cnt +1
    #         json_d['entities'] = label_entities
    #         total_results.append(json_d)
    #         file.write(tokenizer.decode(sent_tokens)+";"+json.dumps(label_entities)+"\n")
    # file.close()
