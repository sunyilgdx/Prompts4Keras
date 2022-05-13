#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'
import numpy
import copy
import argparse
import tensorflow as tf
from tqdm import tqdm
from sklearn import metrics
from bert4keras.backend import keras, K
from bert4keras.optimizers import Adam
from keras.layers import Lambda, Dense
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.models import Model
from utils.dataset import *
from utils.prompt import *


class train_data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern_list, is_pre=True, *args, **kwargs):
        super(train_data_generator, self).__init__(*args, **kwargs)
        self.pattern_list = pattern_list
        self.is_pre = is_pre

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_target_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            for i, pattern in enumerate(self.pattern_list):
                if (self.is_pre):
                    token_ids, segment_ids = tokenizer.encode(first_text=pattern, second_text=text, maxlen=maxlen)
                else:
                    token_ids, segment_ids = tokenizer.encode(first_text=pattern, second_text=text, maxlen=maxlen)
                target_label = 1.0 if i == label else 0.0
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_target_labels.append([target_label])
            if len(batch_token_ids) == self.batch_size * len(self.pattern_list) or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_target_labels = sequence_padding(batch_target_labels)
                yield [batch_token_ids, batch_segment_ids], batch_target_labels
                batch_token_ids, batch_segment_ids, batch_target_labels = [], [], []


class test_data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern="", is_pre=True, *args, **kwargs):
        super(test_data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.is_pre = is_pre

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=self.pattern, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class Evaluator(keras.callbacks.Callback):

    def __init__(self):
        self.best_val_acc = 0.
        self.final_test_acc = 0.

    def on_epoch_end(self, epoch, logs=None):

        val_acc = evaluate(dev_generator_list, dev_data, note="Dev Set")
        if val_acc >= self.best_val_acc:
            test_acc = evaluate(test_generator_list, test_data, note="Test Set")
            self.best_val_acc = val_acc
            # nsp_encoder.save_weights('nsp_encoder.weights')
            self.final_test_acc = test_acc
            print("Val metric: {:.4f}, test metric: {:.4f}".format(val_acc, test_acc))
        else:
            print("Val metric: {:.4f}".format(val_acc))
        print('Best val metric: {:.4f}, final test metric: {:.4f}'.format(self.best_val_acc, self.final_test_acc))
        print()
        print()


def evaluate(data_generator_list, data, note=""):
    print("*******************Start to predict on 【{}】*******************".format(note))
    patterns_logits = [[] for _ in pattern_list]
    for i in tqdm(range(len(data_generator_list)), desc='Pattern i:'):
        # print("Pattern{}".format(i))
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in data_generator:
            outputs = nsp_encoder.predict(x[:2])
            for out in outputs:
                logit_pos = out.T
                patterns_logits[i].append(logit_pos)
                counter += 1

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = []
    for i in range(len(patterns_logits[0])):
        pred = numpy.argmax([logits[i] for logits in patterns_logits])
        preds.append(int(pred))

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    if (dataset.metric == 'Matthews'):
        matthews_corrcoef = metrics.matthews_corrcoef(trues, preds)
        print("Matthews Corrcoef:\t{}".format(matthews_corrcoef))
        print("Confusion Matrix:\n{}".format(confusion_matrix))
        return matthews_corrcoef
    else:
        acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
        macro_f1 = metrics.f1_score(trues, preds, average='macro')
        print("Acc.:\t{:.4f}".format(acc))
        print("Macro F1:\t{:.4f}".format(macro_f1))
        print("Confusion Matrix:\n{}".format(confusion_matrix))
        return acc


def get_nsp_encoder(config_path, checkpoint_path, model='bert', ):
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        model=model,
        with_nsp=True,
        return_keras_model=False
    )
    nsp_output = bert.model.get_layer('NSP-Proba').output
    output = keras.layers.Lambda(lambda x: x[:, 0])(nsp_output)
    encoder = keras.models.Model(bert.model.inputs, output)
    return encoder, bert


def nsp_contrast_loss(y_true, y_pred):

    # Using categorical_crossentropy to realize InfoNCE (softmax with temperature)
    if (args.loss_function == 'Contrastive'):
        y_true_ = K.reshape(y_true, (-1, label_num))
        y_pred_ = K.reshape(y_pred, (-1, label_num))
        y_pred_ = y_pred_ * args.temperature
        loss = K.categorical_crossentropy(y_true_, y_pred_, from_logits=False)

    # Using binary crossentropy
    elif(args.loss_function=='BCE'):
        y_true_ = K.reshape(y_true, (-1, label_num))
        y_pred_ = K.reshape(y_pred, (-1, label_num))
        loss = K.binary_crossentropy(y_true_, y_pred_, from_logits=False)

    # Using categorical_crossentropy to realize softmax
    elif(args.loss_function=='softmax'):
        y_true_ = K.reshape(y_true, (-1, label_num))
        y_pred_ = K.reshape(y_pred, (-1, label_num))
        loss = K.categorical_crossentropy(y_true_, y_pred_, from_logits=False)

    else:
        print("Wrong loss function, use softmax as defualt.")
        y_true_ = K.reshape(y_true, (-1, label_num))
        y_pred_ = K.reshape(y_pred, (-1, label_num))
        loss = K.binary_crossentropy(y_true_, y_pred_, from_logits=False)

    return K.mean(loss)

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
    n_th_set = args.n_th_set  # 1~5
    dataset = Datasets(dataset_name=dataset_name, n_th_set=n_th_set)
    epoch_num = args.epochs

    global pattern_list
    nsp_prompt = NSP_Prompt(dataset_name=dataset_name)
    label_texts = nsp_prompt.label_texts
    template = nsp_prompt.template
    pattern_list = [template.replace("[label]", label) for label in label_texts]
    is_pre = nsp_prompt.is_pre
    label_num = len(label_texts)

    # Load the train/dev/test dataset
    global dev_data, test_data, dev_generator_list, test_generator_list
    train_data = dataset.load_data(dataset.train_path, sample_num=-1, is_shuffle=True)
    train_generator = train_data_generator(pattern_list=pattern_list, is_pre=is_pre, data=train_data,
                                           batch_size=batch_size)

    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True, random_seed=0)
    dev_generator_list = []
    for p in pattern_list:
        dev_generator_list.append(
            test_data_generator(pattern=p, is_pre=is_pre, data=dev_data, batch_size=predict_batch_size))

    test_data = copy.deepcopy(dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True, random_seed=1))
    test_generator_list = []
    for p in pattern_list:
        test_generator_list.append(
            test_data_generator(pattern=p, is_pre=is_pre, data=test_data, batch_size=predict_batch_size))

    # BERT tokenizer
    global tokenizer
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)

    # Load BERT model with NSP head
    global nsp_encoder
    nsp_encoder, bert = get_nsp_encoder(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        model='bert'
    )

    if (args.method == "few-shot"):
        # Training model
        evaluator = Evaluator()
        # Check the model details
        # nsp_encoder.summary()
        nsp_encoder.compile(loss=nsp_contrast_loss, optimizer=Adam(args.learning_rate))
        nsp_encoder.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epoch_num,
            callbacks=[evaluator],
            shuffle=True
        )
        # Save BERT model (Not the nsp_encoder)
        # bert.save_weights_as_checkpoint("./save/bert_model.ckpt")
    else:
        # Zero shot prediction.
        val_acc = evaluate(dev_generator_list, dev_data, note="Dev Set")
        test_acc = evaluate(test_generator_list, test_data, note="Test Set")
        print("Val metric: {:.4f}, test metric: {:.4f}".format(val_acc, test_acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run PET-RoBERTa classification for few-shot or zero-shot.')
    # About datasets
    parser.add_argument('--n_th_set', type=int, choices=[-1, 1, 2, 3, 4, 5], help="Random Sub Dataset", default=1)
    parser.add_argument('--dataset_name', type=str, help="The dowmstream task dataset name", default="SST-2")
    # About model or parameters
    parser.add_argument("--method", type=str, default='zero-shot', choices=['few-shot', 'zero-shot'],
                        help="Scenario for evaluating the model.")
    parser.add_argument("--model_name", type=str, default='roberta_large',
                        choices=['roberta_base', 'roberta_large', 'roberta_large_wiki_books'],
                        help="The model in our code.")
    parser.add_argument("--device", type=str, default='0', help="The device to train model, -1 means CPU.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--predict_batch_size", type=int, default=32, help="Batch size while predicting.")
    parser.add_argument("--loss_function", type=str, default='softmax', choices=['softmax'],
                        help="The loss function used by different tasks.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="The epochs of training.")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length.")
    args = parser.parse_args()
    print("===================================Nth Set: {}===================================".format(args.n_th_set))
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    main()
