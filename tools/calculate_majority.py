#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import metrics
from utils.dataset import *
from utils.prompt import *

if __name__ == "__main__":
    for dataset_name in ['SST-2', 'MR', 'CR', 'MPQA', 'Subj', 'Yahoo', 'AGNews', 'EPRSTMT', 'TNEWS', 'CLSDCP',
                         'IFLYTEK']:

        pet_prompt = PET_Prompt(dataset_name=dataset_name)
        label_texts = pet_prompt.label_texts
        label_num = len(label_texts)
        dataset = Datasets(dataset_name=dataset_name, n_th_set=1)
        test_data = dataset.load_data('.' + dataset.test_path, sample_num=-1, is_shuffle=True, random_seed=1)
        trues = [d[-1] for d in test_data]
        max_majority = -1
        for random_guess in range(label_num):
            preds = [random_guess] * len(trues)
            acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
            max_majority = acc if acc > max_majority else max_majority
        print(max_majority)
