# 作者 ：duty
# 创建时间 ：2022/9/21 11:06 上午
# 文件 ：my_bert_model.py
from collections import OrderedDict

import torch
from transformers import BertModel, BertConfig, BertTokenizer
import os
import gensim
pwd = os.path.dirname(__file__)
father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_bert_model(default_tag, bert_dir, max_seq_len):
    # word2vec_model = gensim.models.Word2Vec.load(father_path+"/nlp_ner_app/model/w2v/word2vec.model")
    ##如果使用默认的bert
    if default_tag:
        config = BertConfig()
        config.max_length = max_seq_len
        my_bert_model = BertModel(config)
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
        config = BertConfig()
        config.max_length = max_seq_len
        my_bert_model = BertModel(config)
        ## 加载新词
        new_words = []
        file = open("./data/new_unk_words", "r", encoding="utf-8")
        for line in file.readlines():
            columns = line.split(",")
            if len(columns) != 2:
                continue
            word = columns[0]
            cnt = int(columns[1])
            if cnt <= 4:
                continue
            new_words.append(word)
        file.close()

        tokenizer.add_tokens(new_words)
        # tokenizer.vocab
        ## 将所有词对应的id写入文件以便后续使用
        old_vocab_size = tokenizer.vocab_size
        old_vocab = {}
        for i, item in enumerate(tokenizer.vocab.items()):
            if item[1]>=old_vocab_size:
                continue
            old_vocab[item[0]] = item[1]
        new_vocab = tokenizer.get_added_vocab()
        new_vocab_id2word = OrderedDict()
        for word, id in new_vocab.items():
            new_vocab_id2word[id] = word
        # new_vocab.update(old_vocab)
        total_vocab = {**old_vocab, **new_vocab}
        # total_vocab.update(old_vocab)
        # total_vocab.update(new_vocab)
        ###有新词加入时要更新词典
        new_words_ids_file = open("./data/new_words_ids", "w")
        # for new_word in new_words:
        for key, value in total_vocab.items():
            new_words_ids_file.write(key+","+str(value)+"\n")
        new_words_ids_file.close()
        my_bert_model.resize_token_embeddings(len(tokenizer))
    return config, tokenizer, my_bert_model