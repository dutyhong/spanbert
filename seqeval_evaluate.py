# 作者 ：duty
# 创建时间 ：2021/3/16 3:38 下午
# 文件 ：seqeval_evaluate.py
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
import re


def seqeval_evaluate(true_seq_labels, pred_seq_labels):
    """
    通过seqeval工具评估模型准确率
    准确率: accuracy = 预测对的元素个数/总的元素个数
    查准率：precision = 预测正确的实体个数 / 预测的实体总个数
    召回率：recall = 预测正确的实体个数 / 标注的实体总个数
    F1值：F1 = 2 *查准率 * 召回率 / (查准率 + 召回率)
    :return:
    """
    # aa = "S_population S_category S_brand B_brand E_brand"
    # bb = "S_population S_category O B_brand E_brand"
    y_trues = []
    y_preds = []
    for true_seq_label, pred_seq_label in zip(true_seq_labels, pred_seq_labels):
        # true_seq_label = re.sub('CLS'|'SEP', 'O', true_seq_label)
        y_true = true_seq_label.replace("_", "-").replace("CLS", "O").replace("SEP", "O").split(" ")
        y_pred = pred_seq_label.replace("_", "-").replace("CLS", "O").replace("SEP", "O").split(" ")
        y_trues.append(y_true)
        y_preds.append(y_pred)
    acc_score = accuracy_score(y_trues, y_preds)
    pre_score = precision_score(y_trues, y_preds)
    rec_score = recall_score(y_trues, y_preds)
    F1_score = f1_score(y_trues, y_preds)
    print("accuary: ", acc_score)
    print("p: ", pre_score)
    print("r: ", rec_score)
    print("f1: ", F1_score)
    print(classification_report(y_trues, y_preds))
    return acc_score, pre_score, rec_score, F1_score


def seqeval_evaluate_customized(ori_sents, true_seq_labels, pred_seq_labels):
    """
    通过seqeval工具评估模型准确率
    准确率: accuracy = 预测对的元素个数/总的元素个数
    查准率：precision = 预测正确的实体个数 / 预测的实体总个数
    召回率：recall = 预测正确的实体个数 / 标注的实体总个数
    F1值：F1 = 2 *查准率 * 召回率 / (查准率 + 召回率)
    B_c E_c = S_m S_c
    :return:
    """
    # aa = "S_population S_category S_brand B_brand E_brand"
    # bb = "S_population S_category O B_brand E_brand"
    y_trues = []
    y_preds = []
    # file2 = open("../../data/bad_cases", "w")
    for true_seq_label, pred_seq_label in zip(true_seq_labels, pred_seq_labels):
        # true_seq_label = re.sub('CLS'|'SEP', 'O', true_seq_label)
        # pred_seq_label = pred_seq_label.replace("S_modifier S_category","B_category E_category")
        pred_seq_label = replace_rules(pred_seq_label)
        y_true = true_seq_label.replace("_", "-").replace("CLS", "O").replace("SEP", "O").split(" ")
        y_pred = pred_seq_label.replace("_", "-").replace("CLS", "O").replace("SEP", "O").split(" ")

        y_trues.append(y_true)
        y_preds.append(y_pred)
    # for ori_sent, y_true, y_pred in zip(ori_sents, y_trues, y_preds):
    #     if " ".join(y_true) != " ".join(y_pred):
    #         file2.write(ori_sent + ";" + " ".join(y_true) + ";" + " ".join(y_pred) + "\n")
    acc_score = accuracy_score(y_trues, y_preds)
    pre_score = precision_score(y_trues, y_preds)
    rec_score = recall_score(y_trues, y_preds)
    F1_score = f1_score(y_trues, y_preds)
    print("accuary: ", acc_score)
    print("p: ", pre_score)
    print("r: ", rec_score)
    print("f1: ", F1_score)
    print(classification_report(y_trues, y_preds))
    return acc_score, pre_score, rec_score, F1_score


def replace_rules(pred_seq_labels):
    """
    模型输出之后的替换规则，比如单独的B I E替换为S
    :return:
    """
    # print(pred_seq_labels)
    pred_seq_labels = pred_seq_labels.split(" ")
    category_list = []
    pos_list = []
    category_list.append("O")
    pos_list.append("O")
    for i, pred_seq_label in enumerate(pred_seq_labels):
        if pred_seq_label == "O":
            category_list.append("O")
            pos_list.append("O")
        else:
            values = pred_seq_label.split("_")
            category = values[1]
            pos = values[0]
            category_list.append(category)
            pos_list.append(pos)
    category_list.append("O")
    pos_list.append("O")
    for i in range(len(pred_seq_labels)):
        if pos_list[i + 1] in ["B", "I", "E"] and category_list[i + 1] != category_list[i] and category_list[i + 1] != \
                category_list[i + 2]:
            new_tag = "S_" + category_list[i + 1]
            pred_seq_labels[i] = new_tag
    ## BS BI类别相同，但是和前后两个类别不一样，或者前一个为E后一个类别必须不同
    for i in range(len(pred_seq_labels) - 1):
        if ((pos_list[i + 1] == "B" and pos_list[i + 2] == "S" and category_list[i + 1] == category_list[i + 2])
            or (pos_list[i + 1] == "B" and pos_list[i + 2] == "I" and category_list[i + 1] == category_list[i + 2])) and \
                (category_list[i] != category_list[i + 1] or pos_list[i]=="E") and category_list[i + 2] != category_list[i + 3]:
            new_tag = "E_" + category_list[i + 1]
            pred_seq_labels[i + 1] = new_tag

    ## BII 类别相同，前后类别不同，或者前一个为E 后一个类别bixu不同
    for i in range(len(pred_seq_labels) - 1):
        if (pos_list[i + 1] == "B" and pos_list[i + 2] == "I" and pos_list[i+3]=="I" and category_list[i + 1] == category_list[i + 2] and category_list[i+2]==category_list[i+3]) and \
            (category_list[i]!=category_list[i+1] or pos_list[i]=="E") and category_list[i+3]!=category_list[i+4]:
            new_tag = "E_" + category_list[i + 3]
            pred_seq_labels[i + 2] = new_tag

    return " ".join(pred_seq_labels)


# if __name__ == "__main__":
def seq_eval():
    aa = ["B_brand B_category S_category S_ip B_size I_size I_size",
          "O S_category S_category S_category S_category","S_brand E_category"]  ##["CLS S_population S_category S_brand B_brand E_brand SEP O","CLS S_category O SEP O O O O"]
    bb = ["S_modifier E_modifier E_category",
          "O S_category E_category O S_category"]  ##["CLS S_population S_category B_brand B_brand S_brand SEP SEP","CLS S_category O SEP SEP SEP SEP SEP"]
    ## 读取测试输出文件
    # file = open("../../data/test_result_v4", "r")
    file = open("./data/bert_span_evaluate_result", "r")

    ori_sents = []
    y_trues = []
    y_preds = []
    for line in file.readlines():
        columns = line.rstrip().split(";")
        if len(columns) != 3:
            continue
        ori_sents.append(columns[0])
        y_true = columns[1]
        y_pred = columns[2]
        y_trues.append(y_true)
        y_preds.append(y_pred)
    # acc, precision, recall, f1 = seqeval_evaluate(y_trues, y_preds)
    acc, precision, recall, f1 = seqeval_evaluate_customized(ori_sents, y_trues, y_preds)
    # find_error_cases()
    file.close()
    # print("ddd")
    return acc, precision, recall, f1
