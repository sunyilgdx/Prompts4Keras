#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/6/30
import csv
import json
import random


class Datasets():

    def __init__(self, dataset_name="", n_th_set=1):
        self.dataset_name = dataset_name
        self.patterns = []
        self.label_num = -1
        self.train_path, self.dev_path, self.test_path = "", "", ""
        self.n_th2seed = {1: 13, 2: 21, 3: 42, 4: 87, 5: 100}

        if (dataset_name in ['EPRSTMT', 'CSLDCP', 'IFLYTEK', 'BUSTM', 'CHID', 'CSL', 'CLUEWSC']):
            self.train_path = r"./datasets/few_clue/{}/train_{}.json".format(dataset_name.lower(), n_th_set - 1)
            self.dev_path = r"./datasets/few_clue/{}/dev_{}.json".format(dataset_name.lower(), n_th_set - 1)
            self.test_path = r"./datasets/few_clue/{}/test_public.json".format(dataset_name.lower())

        elif (dataset_name in ['TNEWS', 'TNEWSK']):
            self.train_path = r"./datasets/few_clue/tnews/train_{}.json".format(n_th_set - 1)
            self.dev_path = r"./datasets/few_clue/tnews/dev_{}.json".format(n_th_set - 1)
            self.test_path = r"./datasets/few_clue/tnews/test_public.json"

        elif (dataset_name in ['SST-2', 'Amazon', 'IMDB', 'DBpedia', 'QQP', 'RTE', 'SNLI', 'SST-B', 'QNLI', 'MRPC']):
            if (n_th_set == -1):
                self.train_path = r"./datasets/{}/train.tsv".format(dataset_name)
                self.dev_path = r"./datasets/{}/dev.tsv".format(dataset_name)
                self.test_path = r"./datasets/{}/dev.tsv".format(dataset_name)
            else:
                self.train_path = r"./datasets/k-shot-10x/{}/16-{}/train.tsv".format(dataset_name,
                                                                                     self.n_th2seed[n_th_set])
                self.dev_path = r"./datasets/k-shot-10x/{}/16-{}/dev.tsv".format(dataset_name, self.n_th2seed[n_th_set])
                self.test_path = r"./datasets/k-shot-10x/{}/16-{}/test.tsv".format(dataset_name, self.n_th2seed[n_th_set])

        elif (dataset_name in ['SST-5', 'MR', 'CR', 'Subj', 'MPQA', 'TREC', 'AGNews', 'Yahoo']):
            if (n_th_set == -1):
                self.train_path = r"./datasets/{}/train.csv".format(dataset_name)
                self.dev_path = r"./datasets/{}/test.csv".format(dataset_name)
                self.test_path = r"./datasets/{}/test.csv".format(dataset_name)
            else:
                self.train_path = r"./datasets/k-shot-10x/{}/16-{}/train.csv".format(dataset_name, self.n_th2seed[n_th_set])
                self.dev_path = r"./datasets/k-shot-10x/{}/16-{}/dev.csv".format(dataset_name, self.n_th2seed[n_th_set])
                self.test_path = r"./datasets/k-shot-10x/{}/16-{}/test.csv".format(dataset_name, self.n_th2seed[n_th_set])

        elif (dataset_name in ['MNLI-mm', 'MNLI-m']):
            self.label_num = 3
            if (n_th_set == -1):
                self.train_path = r"./datasets/MNLI/train.tsv"
                self.dev_path = r"./datasets/MNLI/dev_matched.tsv"
                self.test_path = r"./datasets/MNLI/dev_matched.tsv"
            else:
                suffix = "matched" if dataset_name == "MNLI-mm" else "mismatched"
                self.train_path = r"./datasets/k-shot-10x/MNLI/16-{}/train_{}.csv".format(self.n_th2seed[n_th_set],
                                                                                          suffix)
                self.dev_path = r"./datasets/k-shot-10x/MNLI/16-{}/dev_{}.csv".format(self.n_th2seed[n_th_set], suffix)
                self.test_path = r"./datasets/k-shot-10x/MNLI/16-{}/test_{}.csv".format(self.n_th2seed[n_th_set],
                                                                                        suffix)

        elif (dataset_name in ['OCNLI']):
            self.label_num = 3
            if (n_th_set == -1):
                self.train_path = r"./datasets/OCNLI/train.50k.json"
                self.dev_path = r"./datasets/OCNLI/dev.json"
                self.test_path = r"./datasets/OCNLI/dev.json"
            else:
                self.train_path = r"./datasets/few_clue/{}/train_{}.json".format(dataset_name.lower(), n_th_set - 1)
                self.dev_path = r"./datasets/few_clue/{}/dev_{}.json".format(dataset_name.lower(), n_th_set - 1)
                self.test_path = r"./datasets/few_clue/{}/test_public.json".format(dataset_name.lower())


        elif (dataset_name == "duel2.0"):
            self.train_path = r"./datasets/DuEL 2.0/train.json"
            self.dev_path = r"./datasets/DuEL 2.0/dev.json"
            self.test_path = r"./datasets/DuEL 2.0/test.json"
            self.kb_path = r"./datasets/DuEL 2.0/kb.json"
            self.type_en2zh = {'Event': '事件活动', 'Person': '人物', 'Work': '作品', 'Location': '区域场所',
                               'Time&Calendar': '时间历法', 'Brand': '品牌', 'Natural&Geography': '自然地理',
                               'Game': '游戏', 'Biological': '生物', 'Medicine': '药物', 'Food': '食物',
                               'Software': '软件', 'Vehicle': '车辆', 'Website': '网站平台', 'Disease&Symptom': '疾病症状',
                               'Organization': '组织机构', 'Awards': '奖项', 'Education': '教育', 'Culture': '文化',
                               'Constellation': '星座', 'Law&Regulation': '法律法规', 'VirtualThings': '虚拟事物',
                               'Diagnosis&Treatment': '诊断治疗方法', 'Other': '其他'}
            self.type_list = ['Event', 'Person', 'Work', 'Location', 'Time&Calendar', 'Brand', 'Natural&Geography',
                              'Game', 'Biological', 'Medicine', 'Food', 'Software', 'Vehicle', 'Website',
                              'Disease&Symptom', 'Organization', 'Awards', 'Education', 'Culture', 'Constellation',
                              'Law&Regulation', 'VirtualThings', 'Diagnosis&Treatment', 'Other']

        # The metric of dataset
        if (dataset_name in ['SST-2', 'SST-5', 'MR', 'CR', 'Subj', 'MPQA', 'TREC', 'Amazon', 'IMDB', 'AGNews', 'Yahoo',
                             'DBpedia', 'MNLI-m', 'MNLI-mm', 'SNLI', 'QNLI', 'RTE',
                             'EPRSTMT', 'TNEWS', 'TNEWSK', 'CSLDCP', 'IFLYTEK', 'OCNLI', 'BUSTM', 'CHID', 'CSL',
                             'CLUEWSC']):
            self.metric = 'Acc'
        elif (dataset_name in ['MRPC', 'QQP']):
            self.metric = 'F1'

    def load_data(self, filename, sample_num=-1, is_shuffle=False, random_seed=0):
        D = []

        if (self.dataset_name == "EPRSTMT"):
            text2label = {"Positive": 1, "Negative": 0}
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['sentence']
                    label_text = json.loads(l)['label']
                    label_id = text2label[label_text]
                    D.append((content, int(label_id)))

        elif (self.dataset_name == "TNEWS"):
            label2label = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10,
                           113: 11, 114: 12, 115: 13, 116: 14}
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text = json.loads(l)['sentence']
                    label = json.loads(l)['label']
                    D.append((text, label2label[label]))

        elif (self.dataset_name == "TNEWSK"):
            label2label = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10,
                           113: 11, 114: 12, 115: 13, 116: 14}
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text = json.loads(l)['sentence']
                    keywords = json.loads(l)['keywords']
                    label = json.loads(l)['label']
                    D.append((text + ' ' + keywords, label2label[label]))

        elif (self.dataset_name == "CSLDCP"):
            text2id = {"材料科学与工程": 0, "作物学": 1, "口腔医学": 2, "药学": 3, "教育学": 4, "水利工程": 5, "理论经济学": 6, "食品科学与工程": 7,
                       "畜牧学/兽医学": 8, "体育学": 9, "核科学与技术": 10, "力学": 11, "园艺学": 12, "水产": 13, "法学": 14,
                       "地质学/地质资源与地质工程": 15, "石油与天然气工程": 16, "农林经济管理": 17, "信息与通信工程": 18, "图书馆、情报与档案管理": 19, "政治学": 20,
                       "电气工程": 21, "海洋科学": 22, "民族学": 23, "航空宇航科学与技术": 24, "化学/化学工程与技术": 25, "哲学": 26, "公共卫生与预防医学": 27,
                       "艺术学": 28, "农业工程": 29, "船舶与海洋工程": 30, "计算机科学与技术": 31, "冶金工程": 32, "交通运输工程": 33, "动力工程及工程热物理": 34,
                       "纺织科学与工程": 35, "建筑学": 36, "环境科学与工程": 37, "公共管理": 38, "数学": 39, "物理学": 40, "林学/林业工程": 41,
                       "心理学": 42, "历史学": 43, "工商管理": 44, "应用经济学": 45, "中医学/中药学": 46, "天文学": 47, "机械工程": 48, "土木工程": 49,
                       "光学工程": 50, "地理学": 51, "农业资源利用": 52, "生物学/生物科学与工程": 53, "兵器科学与技术": 54, "矿业工程": 55, "大气科学": 56,
                       "基础医学/临床医学": 57, "电子科学与技术": 58, "测绘科学与技术": 59, "控制科学与工程": 60, "军事学": 61, "中国语言文学": 62,
                       "新闻传播学": 63, "社会学": 64, "地球物理学": 65, "植物保护": 66,
                       }
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['content']
                    label_text = json.loads(l)['label']
                    label_id = text2id[label_text]
                    D.append((content, int(label_id)))

        elif (self.dataset_name == "IFLYTEK"):

            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text = json.loads(l)['sentence']
                    label = json.loads(l)['label']
                    D.append((text, int(label)))

        elif (self.dataset_name == "OCNLI"):
            label_text2label_id = {"entailment": 2, "contradiction": 0, "neutral": 1}
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    rows = json.loads(l)
                    sentence1 = rows['sentence1']
                    sentence2 = rows['sentence2']
                    label_text = rows['label']
                    if (label_text not in label_text2label_id):
                        continue
                    label = int(label_text2label_id[label_text])
                    D.append((sentence1, sentence2, int(label)))

        elif (self.dataset_name == "bustm"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    sentence1 = json.loads(l)['sentence1']
                    sentence2 = json.loads(l)['sentence2']
                    label = json.loads(l)['label']
                    text = "{}[SEP]{}".format(sentence1, sentence2)
                    D.append((text, int(label)))

        elif (self.dataset_name == "chid"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['content']
                    candidates = json.loads(l)['candidates']
                    label = json.loads(l)['answer']
                    D.append((content, int(label), candidates))

        elif (self.dataset_name == "csl"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['abst']
                    keywords = json.loads(l)['keyword']
                    label = json.loads(l)['label']
                    D.append((content + "[SEP]" + ",".join(keywords), int(label)))

        elif (self.dataset_name == "cluewsc"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    target = json.loads(l)['target']
                    span1_text = target['span1_text']
                    span2_text = target['span2_text']
                    span1_index = target['span1_index']
                    span2_index = target['span2_index']
                    text = json.loads(l)['text']
                    label = json.loads(l)['label']
                    label = self.text2id[label]
                    D.append((text, int(label), span1_text, span2_text, span1_index, span2_index))

        elif (self.dataset_name == "duel2.0"):
            with open(filename, encoding='utf-8')as f:
                for l in f:
                    D.append(json.loads(l))

        if (self.dataset_name in ['SST-2']):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text = rows[-2]
                    label = rows[-1]
                    D.append((text, int(label)))

        elif (self.dataset_name in ['SST-5', 'MR', 'CR', 'Subj', 'MPQA']):
            with open(filename, encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    text = row[1]
                    label = row[0]
                    D.append((text, int(label)))

        elif (self.dataset_name in ['AGNews']):
            with open(filename, encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    headline = row[-2]
                    body = row[-1]
                    label = row[0]
                    D.append((headline + " " + body, int(label) - 1))

        elif (self.dataset_name in ['Yahoo']):
            with open(filename, encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    title = row[1]
                    content = row[2]
                    answer = row[3]
                    label = row[0]
                    D.append(("{} {} {}".format(title, content, answer), int(label) - 1))

        elif (self.dataset_name in ['MNLI-m', 'MNLI-mm', 'SNLI']):
            text2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-8]
                    text_b = rows[-7]
                    label = rows[-1]
                    D.append((text_a, text_b, text2id[label]))

        elif (self.dataset_name in ['QQP', 'WNLI']):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    label = rows[-1]
                    D.append((text_a, text_b, int(label)))

        elif (self.dataset_name in ['QNLI', 'RTE']):
            text2id = {"entailment": 1, "not_entailment": 0}
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-3]
                    text_b = rows[-2]
                    label = rows[-1]
                    D.append((text_a, text_b, text2id[label]))

        elif (self.dataset_name in ['MRPC']):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text_a = rows[-1]
                    text_b = rows[-2]
                    label = rows[0]
                    D.append((text_a, text_b, 1 - int(label)))

        # Shuffle the dataset.
        if (is_shuffle):
            random.seed(random_seed)
            random.shuffle(D)

        # Set the number of samples.
        if (sample_num == -1):
            # -1 for all the samples
            return D
        else:
            return D[:sample_num]

    # Load the Knowledge Base for DuEL2.0.
    def load_kb(self, filename):
        kb_list = []
        mention2id = {}
        id2data = {}
        id2type = {}
        with open(filename, "r", encoding='utf-8') as kb_file:
            for line in kb_file.readlines():
                k = json.loads(line)
                kb_list.append(k)
                subject_id = k["subject_id"]
                alias = k["alias"]
                data = k["data"]
                type = k["type"]
                id2data[subject_id] = data
                id2type[subject_id] = type
                subject = k["subject"]
                if (subject not in alias):
                    alias.append(subject)

                for alia in alias:
                    if (alia not in mention2id):
                        mention2id[alia] = set()
                        mention2id[alia].add(subject_id)
                    else:
                        mention2id[alia].add(subject_id)
        return kb_list, mention2id, id2data, id2type


class Models():

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.config_path, self.checkpoint_path, self.dict_path = "", "", ""

        model_name_map = {'bert_base': 'google-bert-cased-base',
                          'bert_large': 'google-bert-cased-large',
                          'bert_large_mnli': 'google-bert-cased-large-mnli',
                          'bert_large_mixr': 'google-bert-cased-large-mixr',
                          'roberta_base': 'fairseq-roberta-base',
                          'roberta_large': 'fairseq-roberta-large',
                          'roberta_large_mnli': 'fairseq-roberta-large-mnli',
                          'roberta_large_wiki_books': 'fairseq-roberta-large-wiki-books',
                          'chinese_bert_base': "uer-mixed-bert-base",
                          'chinese_bert_base_ocnli': "uer-mixed-bert-base-ocnli",
                           }
        try:
            model_name = model_name_map[model_name]
        except:
            print("Unexpected model name.")

        # English BERT or RoBERTa
        if (model_name == 'google-bert-cased-large'):
            self.config_path = './models/cased_L-24_H-1024_A-16/bert_config.json'
            self.checkpoint_path = './models/cased_L-24_H-1024_A-16/bert_model.ckpt'
            self.dict_path = './models/cased_L-24_H-1024_A-16/vocab.txt'

        elif (model_name == 'google-bert-cased-large-mnli'):
            self.config_path = './models/cased_L-24_H-1024_A-16_mnli/bert_config.json'
            self.checkpoint_path = './models/cased_L-24_H-1024_A-16_mnli/nli_bert.weights'
            self.dict_path = './models/cased_L-24_H-1024_A-16_mnli/vocab.txt'

        elif (model_name == 'google-bert-cased-large-mixr'):
            self.config_path = './models/cased_L-24_H-1024_A-16_mixr/bert_config.json'
            self.checkpoint_path = './models/cased_L-24_H-1024_A-16_mixr/model.ckpt-3000000'
            self.dict_path = './models/cased_L-24_H-1024_A-16_mixr/vocab.txt'

        elif (model_name == 'fairseq-roberta-large'):
            self.config_path = r'./models/roberta_large_fairseq/bert_config.json'
            self.checkpoint_path = r'./models/roberta_large_fairseq/roberta_large.ckpt'
            self.merges_file = r'./models/roberta_large_fairseq/merges.txt'
            self.dict_path = r'./models/roberta_large_fairseq/vocab.json'

        elif (model_name == 'fairseq-roberta-large-mnli'):
            self.config_path = r'./models/roberta_large_fairseq_mnli/bert_config.json'
            self.checkpoint_path = r'./models/roberta_large_fairseq_mnli/roberta_large_mnli.ckpt'
            self.merges_file = r'./models/roberta_large_fairseq_mnli/merges.txt'
            self.dict_path = r'./models/roberta_large_fairseq_mnli/vocab.json'

        elif (model_name == 'fairseq-roberta-large-wiki-books'):
            self.config_path = r'./models/roberta_large_fairseq_wiki_books/bert_config.json'
            self.checkpoint_path = r'./models/roberta_large_fairseq_wiki_books/roberta_large.ckpt'
            self.merges_file = r'./models/roberta_large_fairseq_wiki_books/merges.txt'
            self.dict_path = r'./models/roberta_large_fairseq_wiki_books/vocab.json'


        # Chinese BERT
        elif (model_name == "uer-mixed-bert-base"):
            self.config_path = './models/uer_mixed_corpus_bert_base/bert_config.json'
            self.checkpoint_path = './models/uer_mixed_corpus_bert_base/bert_model.ckpt'
            self.dict_path = './models/uer_mixed_corpus_bert_base/vocab.txt'

        elif (model_name == "uer-mixed-bert-base-ocnli"):
            self.config_path = './models/uer_mixed_corpus_bert_base_ocnli/bert_config.json'
            self.checkpoint_path = './models/uer_mixed_corpus_bert_base_ocnli/bert_model.ckpt'
            self.dict_path = './models/uer_mixed_corpus_bert_base_ocnli/vocab.txt'


def read_labels(label_file_path):
    labels_text = []
    text2id = {}
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            label = line.strip('\n')
            labels_text.append(label)
            text2id[label] = index
    return labels_text, text2id
