#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'
import numpy as np
import argparse
import re, sys
from tqdm import tqdm
from sklearn import metrics
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from utils.dataset import *
from utils.prompt import *


def pre_tokenize(text:str):
    """
    Recognize special tokens like '[label]', [sentence1]
    """
    special_tokens = ['[label]', '[sentence1]']
    special_tokens_num = 0
    special_token = ""
    for token in special_tokens:
        if (token in text):
            special_token = token
            special_tokens_num += 1
    if (special_tokens_num >= 2):
        print("Error: There are more than 1 special token in the text, can't not split.")
        sys.exit(0)
    elif (special_tokens_num <= 0):
        print("Error: There is no special token in the text, can't not split.")
        sys.exit(0)
    left_text, right_text = text.split(special_token)
    return left_text, special_token, right_text


def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


class data_generator(DataGenerator):

    def __init__(self, template, is_pre, is_train, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.template = template
        self.is_pre = is_pre
        self.is_train = is_train

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids, batch_mask_idxs, batch_label_token_ids = [], [], [], [], []
        for is_end, (text, label) in self.sample(random):
            label_token_ids = label_token_ids_list[label]
            text = self.template.replace("[sentence1]", text)
            prompt = self.template.replace("[sentence1]", "").replace("[label]", "")
            prompt_len = len(tokenizer.encode(prompt)[0][1:-1]) + len(label_token_ids)
            text_left, text_mask, text_right = pre_tokenize(text)

            token_ids_left = tokenizer.encode(text_left, maxlen=maxlen if self.is_pre else maxlen - prompt_len)[0][:-1]
            token_ids_right = tokenizer.encode(text_right, maxlen=maxlen - prompt_len if self.is_pre else maxlen)[0][1:]
            mask_idxs = [len(token_ids_left) + i for i in range(len(label_token_ids))]
            masked_token_ids = token_ids_left + [tokenizer._token_mask_id] * len(label_token_ids) + token_ids_right
            unmasked_token_ids = token_ids_left + label_token_ids + token_ids_right
            assert len(masked_token_ids) == len(unmasked_token_ids)
            segment_ids = [0] * len(masked_token_ids)
            source_ids, target_ids = masked_token_ids[:], unmasked_token_ids[:]

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            batch_mask_idxs.append(mask_idxs)
            batch_label_token_ids.append(label_token_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                if (self.is_train):
                    yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                else:
                    yield [batch_token_ids, batch_segment_ids, batch_mask_idxs], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
                batch_mask_idxs, batch_label_token_ids = [], []


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.
        self.final_test_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(dev_generator, dev_data, 'dev')

        if val_acc >= self.best_val_acc:
            test_acc = evaluate(test_generator, test_data, 'test')
            self.best_val_acc = val_acc
            # model.save_weights('best_model.weights')
            self.final_test_acc = test_acc
            print("Val metric: {:.4f}, test metric: {:.4f}".format(val_acc, test_acc))
        else:
            print("Val metric: {:.4f}".format(val_acc))
        print('Best val metric: {:.4f}, final test metric: {:.4f}'.format(self.best_val_acc, self.final_test_acc))
        print()
        print()

def evaluate(data_generator, data, note=""):
    print("\n*******************Start to Few-Shot predict on 【{}】*******************".format(note), flush=True)
    trues, preds = [], []
    logits = []
    trues = [d[-1] for d in data]
    label_ids = np.array(label_token_ids_list)
    for inputs, _ in tqdm(data_generator):
        x_true, batch_mask_idxs = inputs[:2], inputs[2]
        batch_preds = model.predict(x_true)
        mask_preds = []
        for mask_idxs, pred in zip(batch_mask_idxs, batch_preds):
            mask_preds.append(pred[mask_idxs])
        mask_preds = np.array(mask_preds)
        # This requires all Labels to be the same length
        yyy = label_ids[:, 0]
        y_preds = mask_preds[:, 0, label_ids[:, 0]]
        for i in range(1, len(label_ids[0])):
            y_preds = y_preds * mask_preds[:, i, label_ids[:, i]]
        y_preds = y_preds.argmax(axis=1)
        preds += y_preds.tolist()

    # for logit in logits: print(logit)

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    if (dataset.metric == 'Matthews'):
        matthews_corrcoef = metrics.matthews_corrcoef(trues, preds)
        print("Matthews Corrcoef:\n{}".format(matthews_corrcoef), flush=True)
        print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
        return matthews_corrcoef
    else:
        acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
        macro_f1 = metrics.f1_score(trues, preds, average='macro')
        print("Acc.:\t{:.4f}".format(acc), flush=True)
        print("Macro F1:\t{:.4f}".format(macro_f1), flush=True)
        print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
        return acc


class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    global maxlen
    dataset_name = args.dataset_name
    maxlen = args.max_len  # The max length 128 is used in our paper
    batch_size = args.batch_size
    predict_batch_size = args.predict_batch_size
    model_name = args.model_name

    # Load model and dataset class
    global dataset, label_num, epoch_num
    bert_model = Models(model_name=model_name)
    n_th_set = args.n_th_set
    dataset = Datasets(dataset_name=dataset_name, n_th_set=n_th_set)
    epoch_num = args.epochs

    # Build BERT model---------------------------------------------------------------------
    global tokenizer, label_token_ids_list
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
    pet_prompt = PET_Prompt(dataset_name=dataset_name)
    is_pre = pet_prompt.is_pre
    label_texts = pet_prompt.label_texts
    template = pet_prompt.template
    # Check if the label text are in Chinese
    if (is_chinese(label_texts[0])):
        label_tokens_list = [[t for t in label_text] for label_text in label_texts]
    else:
        label_tokens_list = [label_text.split(' ') if ' ' in label_texts else [label_text] for label_text in label_texts]
    # Try to turn the label text to token_id
    try:
        label_token_ids_list = [[tokenizer.token_to_id(t.lower()) for t in label_tokens] for label_tokens in label_tokens_list]
    except:
        print("The labels can't be encodered by the tokenizer directly.")
        return

    # Load the train/dev/test dataset
    global dev_data, test_data, dev_generator, test_generator
    train_data = dataset.load_data(dataset.train_path, sample_num=-1, is_shuffle=True)
    train_generator = data_generator(data=train_data, template=template, is_pre=is_pre, is_train=True, batch_size=batch_size)

    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True, random_seed=0)
    dev_generator = data_generator(data=dev_data, template=template, is_pre=is_pre, is_train=False, batch_size=predict_batch_size)

    test_data = dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True, random_seed=1)
    test_generator = data_generator(data=test_data, template=template, is_pre=is_pre, is_train=False, batch_size=predict_batch_size)

    # Load model from checkpoint with mlm head
    global model
    model = build_transformer_model(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        with_mlm=True
    )

    # Model
    y_in = keras.layers.Input(shape=(None,))
    outputs = CrossEntropy(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    if (args.method == "few-shot"):
        # Training model for few-shot
        train_model.compile(optimizer=Adam(args.learning_rate))
        # train_model.summary()
        evaluator = Evaluator()
        train_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epoch_num,
            callbacks=[evaluator]
        )
    else:
        # Zero shot prediction.
        val_acc = evaluate(dev_generator, dev_data, 'dev')
        test_acc = evaluate(test_generator, test_data, 'test')
        print("Val metric: {:.4f}, test metric: {:.4f}".format(val_acc, test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PET-BERT classification for few-shot or zero-shot.')
    # About datasets
    parser.add_argument('--n_th_set', type=int, choices=[-1, 1, 2, 3, 4, 5], help="Random Sub Dataset", default=1)
    parser.add_argument('--dataset_name', type=str, help="The dowmstream task dataset name", default="SST-2")
    # About model or parameters
    parser.add_argument("--method", type=str, default='zero-shot', choices=['few-shot', 'zero-shot'],
                        help="Scenario for evaluating the model.")
    parser.add_argument("--model_name", type=str, default='bert_large',
                        choices=['bert_base', 'bert_large', 'chinese_bert_base', 'bert_large_mixr'],
                        help="The model in our code.")
    parser.add_argument("--device", type=str, default='0', help="The device to train model, -1 means CPU.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--predict_batch_size", type=int, default=32, help="Batch size while predicting.")
    parser.add_argument("--loss_function", type=str, default='softmax', choices=['softmax'],
                        help="The loss function used by different tasks.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, choices=[1e-5, 2e-5, 3e-5], help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="The epochs of training.")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length.")
    args = parser.parse_args()
    print("===================================Nth Set: {}===================================".format(args.n_th_set))
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    main()
