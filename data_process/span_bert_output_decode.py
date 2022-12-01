# 作者 ：duty
# 创建时间 ：2021/5/10 9:19 下午
# 文件 ：span_bert_output_decode.py
import torch


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S