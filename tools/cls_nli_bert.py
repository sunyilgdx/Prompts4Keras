#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
import numpy as np
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


class data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_target_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):

            token_ids, segment_ids = tokenizer.encode(first_text=text1, second_text=text2, maxlen=maxlen)
            target_label = label
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_target_labels.append([target_label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_target_labels = sequence_padding(batch_target_labels)
                yield [batch_token_ids, batch_segment_ids], batch_target_labels
                batch_token_ids, batch_segment_ids, batch_target_labels = [], [], []


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.
        self.final_test_acc = 0.

    def on_epoch_end(self, epoch, logs=None):

        print("\nEpoch:\t{}".format(epoch + 1))
        val_acc = evaluate(dev_generator, dev_data, 'dev')
        print("Val metric: {:.4f}".format(val_acc))
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            print("Saving best weights...".format(val_acc))
            bert.save_weights_as_checkpoint('nli_bert.weights')
            cls_encoder.save_weights('cls_nli_bert_encoder.weights')
            self.best_epoch = epoch + 1
        if (epoch >= epoch_num - 1):
            print("Loading best weights...".format(val_acc))
            cls_encoder.load_weights('cls_nli_bert_encoder.weights')
            test_acc = evaluate(test_generator, test_data, 'test')
            self.final_test_acc = test_acc
            print("Best Epoch:\t{}".format(self.best_epoch))
            print("Best val metric: {:.4f}, final test metric: {:.4f}".format(self.best_val_acc, self.final_test_acc))
        print()
        print()


def evaluate(data_generator, data, note=""):
    print("*******************Start to predict on 【{}】*******************".format(note))
    preds = []
    logits = []
    counter = 0
    for (x, _) in tqdm(data_generator):
        outputs = cls_encoder.predict(x[:2])
        for out in outputs:
            logit_labels = out.T
            logits.append(logit_labels.tolist())
            preds.append(np.argmax(logit_labels))
            counter += 1

    # Evaluate the results
    trues = [d[-1] for d in data]

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    if (dataset.metric == 'Matthews'):
        matthews_corrcoef = metrics.matthews_corrcoef(trues, preds)
        print("Matthews Corrcoef:{}".format(matthews_corrcoef))
        print("Confusion Matrix:{}".format(confusion_matrix))
        return matthews_corrcoef
    else:
        acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
        macro_f1 = metrics.f1_score(trues, preds, average='macro')
        print("Acc.:\t{:.4f}".format(acc))
        print("Macro F1:\t{:.4f}".format(macro_f1))
        print("Confusion Matrix:{}".format(confusion_matrix))
        return acc


def get_cls_encoder(config_path,
                    checkpoint_path,
                    model='bert', ):
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        model=model,
        with_pool=False,
        return_keras_model=False,
    )
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = Dense(
        units=label_num,
        activation='softmax',
        kernel_initializer=bert.initializer,
        name='Softmax-Probas'
    )(output)

    encoder = keras.models.Model(bert.inputs, output)
    return encoder, bert


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
    label_num = dataset.label_num
    print("label_num:\t{}".format(label_num))
    epoch_num = args.epochs

    # Load the train/dev/test dataset
    global dev_data, test_data, dev_generator, test_generator
    train_data = dataset.load_data(dataset.train_path, sample_num=-1, is_shuffle=True)
    print("Len train_data:\t{}".format(len(train_data)))
    train_generator = data_generator(data=train_data, batch_size=batch_size)

    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True, random_seed=0)
    dev_generator = data_generator(data=dev_data, batch_size=predict_batch_size)

    test_data = dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True, random_seed=1)
    test_generator = data_generator(data=test_data, batch_size=predict_batch_size)

    # Build RoBERTa model
    global tokenizer
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)

    # Load BERT model
    global cls_encoder, bert
    cls_encoder, bert = get_cls_encoder(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        model='bert'
    )

    if (args.method == "few-shot"):
        # Training model
        evaluator = Evaluator()
        # cls_encoder.summary()
        cls_encoder.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(args.learning_rate))
        cls_encoder.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epoch_num,
            callbacks=[evaluator],
            shuffle=True
        )
        # Save BERT model (Not the cls_encoder)
        # bert.save_weights_as_checkpoint("./save/bert_model.ckpt")
    else:
        # Zero shot prediction.
        val_acc = evaluate(dev_generator, dev_data, note="Dev Set")
        test_acc = evaluate(test_generator, test_data, note="Test Set")
        print("Val metric: {:.4f}, test metric: {:.4f}".format(val_acc, test_acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run NLI fine-tuning to generate NLI-BERT.')
    # About datasets
    parser.add_argument('--n_th_set', type=int, choices=[-1], help="Random Sub Dataset. -1 means full dataset", default=1)
    parser.add_argument('--dataset_name', type=str, help="The dowmstream task dataset name", default="MNLI",
                        choices=['OCNLI', 'MNLI'])
    # About model or parameters
    parser.add_argument("--method", type=str, default='few-shot', choices=['few-shot'],
                        help="Scenario for evaluating the model.")
    parser.add_argument("--model_name", type=str, default='bert_large',
                        choices=['bert_large', 'chinese_bert_base'], help="The model in our code.")
    parser.add_argument("--device", type=str, default='0', help="The device to train model, -1 means CPU.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--predict_batch_size", type=int, default=32, help="Batch size while predicting.")
    parser.add_argument("--loss_function", type=str, default='softmax', choices=['softmax'],
                        help="The loss function used by NLI tasks.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="The epochs of training.")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length.")
    args = parser.parse_args()
    print("===================================Nth Set: {}===================================".format(args.n_th_set))
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    main()
