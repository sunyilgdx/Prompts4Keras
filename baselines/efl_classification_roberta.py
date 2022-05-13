#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'
import numpy
import copy
import argparse
from tqdm import tqdm
from sklearn import metrics
from bert4keras.backend import keras, K
from bert4keras.optimizers import Adam
from keras.layers import Lambda, Dense
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils.dataset import *
from utils.prompt import *
from utils.bpe_tokenization import *


def is_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


class train_data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern_list, is_pre=True, *args, **kwargs):
        super(train_data_generator, self).__init__(*args, **kwargs)
        self.pattern_list = pattern_list
        self.is_pre = is_pre

    def __iter__(self, random=False):
        batch_token_ids, batch_target_labels, batch_custom_position_ids = [], [], []
        for is_end, (text, pattern, target_label) in self.sample(random):
            if (self.is_pre):
                token_ids, _ = tokenizer.encode(first_text=pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, _ = tokenizer.encode(first_text=text, second_text=pattern, maxlen=maxlen)
            custom_position_ids = [2 + i for i in range(len(token_ids))]
            batch_token_ids.append(token_ids)
            batch_target_labels.append([target_label])
            batch_custom_position_ids.append(custom_position_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_target_labels = sequence_padding(batch_target_labels)
                batch_custom_position_ids = sequence_padding(batch_custom_position_ids, value=1)
                yield [batch_token_ids, batch_custom_position_ids], batch_target_labels
                batch_token_ids, batch_target_labels, batch_custom_position_ids = [], [], []


class test_data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern="", is_pre=True, *args, **kwargs):
        super(test_data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.is_pre = is_pre

    def __iter__(self, random=False):
        batch_token_ids, batch_custom_position_ids = [], []
        for is_end, (text, label) in self.sample(random):
            if (self.is_pre):
                token_ids, _ = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, _ = tokenizer.encode(first_text=text, second_text=self.pattern, maxlen=maxlen)
            source_ids = token_ids[:]
            custom_position_ids = [2 + i for i in range(len(token_ids))]
            batch_token_ids.append(source_ids)
            batch_custom_position_ids.append(custom_position_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_custom_position_ids = sequence_padding(batch_custom_position_ids, value=1)
                yield [batch_token_ids, batch_custom_position_ids], None
                batch_token_ids, batch_custom_position_ids = [], []


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
    print("*******************Start to Few-Shot predict on 【{}】*******************".format(note))
    patterns_logits = [[] for _ in pattern_list]
    for i in range(len(data_generator_list)):
        print("Pattern{}".format(i))
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in tqdm(data_generator):
            outputs = efl_encoder.predict(x[:2])
            for out in outputs:
                logit_pos = out.T[1]
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


def get_efl_encoder(config_path,
                    checkpoint_path,
                    model='bert', ):
    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        model=model,
        with_pool=False,
        return_keras_model=False,
        custom_position_ids=True,
        segment_vocab_size=0  # RoBERTa don't have the segment embeddings (token type embeddings)
    )
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = Dense(
        units=label_num,
        activation='softmax',
        kernel_initializer=bert.initializer,
        name='Softmax-Probas'
    )(output)

    encoder = keras.models.Model(bert.model.inputs, output)
    return encoder, bert


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    global maxlen
    dataset_name = args.dataset_name
    maxlen = args.max_len  # The max length 128 is used in our paper
    batch_size = args.batch_size
    predict_batch_size = args.predict_batch_size
    if (args.with_nli_pretrain == True):
        model_name = args.nli_model_name
    else:
        model_name = args.model_name

    # Load model and dataset class
    global dataset, label_num, epoch_num
    bert_model = Models(model_name=model_name)
    n_th_set = args.n_th_set  # 1~5
    dataset = Datasets(dataset_name=dataset_name, n_th_set=n_th_set)
    epoch_num = args.epochs

    global tokenizer, label_token_ids_list, token_mask_id, pattern_list
    tokenizer = RobertaTokenizer(vocab_file=bert_model.dict_path, merges_file=bert_model.merges_file,
                                 lowercase=True, add_prefix_space=True)
    token_mask_id = tokenizer.token_to_id('<mask>')
    nsp_prompt = NSP_Prompt(dataset_name=dataset_name)
    label_texts = nsp_prompt.label_texts
    template = nsp_prompt.template
    pattern_list = [template.replace("[label]", label) for label in label_texts]
    is_pre = nsp_prompt.is_pre
    label_num = len(label_texts)

    # Check if the label text are in Chinese
    if (is_chinese(label_texts[0])):
        label_tokens_list = [[t for t in label_text] for label_text in label_texts]
    else:
        label_tokens_list = [label_text.split(' ') if ' ' in label_texts else [label_text] for label_text in
                             label_texts]
    # Try to turn the label text to token_id
    try:
        label_token_ids_list = [[tokenizer.token_to_id('Ġ' + t.lower()) for t in label_tokens] for label_tokens in
                                label_tokens_list]
    except:
        print("The labels can't be encodered by the tokenizer directly.")
        return

    # Load the train/dev/test dataset
    global dev_data, test_data, dev_generator_list, test_generator_list
    train_data = dataset.load_data(dataset.train_path, sample_num=-1, is_shuffle=True)
    target_train_data = []
    for text, label in train_data:
        for j, pattern in enumerate(pattern_list):
            target_label = 1 if j == label else 0
            target_train_data.append((text, pattern, target_label))
    random.shuffle(target_train_data)
    train_generator = train_data_generator(pattern_list=pattern_list, is_pre=is_pre, data=target_train_data,
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

    # Load RoBERTa model
    global efl_encoder, bert
    efl_encoder, bert = get_efl_encoder(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        model='bert'
    )

    if (args.method == "few-shot"):
        # Training model
        evaluator = Evaluator()
        # efl_encoder.summary()
        efl_encoder.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(args.learning_rate))
        efl_encoder.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epoch_num,
            callbacks=[evaluator],
            shuffle=True
        )
        # Save BERT model (Not the efl_encoder)
        # bert.save_weights_as_checkpoint("./save/bert_model.ckpt")
    else:
        # Zero shot prediction.
        val_acc = evaluate(dev_generator_list, dev_data, note="Dev Set")
        test_acc = evaluate(test_generator_list, test_data, note="Test Set")
        print("Val metric: {:.4f}, test metric: {:.4f}".format(val_acc, test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run EFL-RoBERTa classification for few-shot.')
    # About datasets
    parser.add_argument('--n_th_set', type=int, choices=[-1, 1, 2, 3, 4, 5], help="Random Sub Dataset", default=1)
    parser.add_argument('--dataset_name', type=str, help="The dowmstream task dataset name", default="SST-2")
    # About model or parameters
    parser.add_argument("--method", type=str, default='few-shot', choices=['few-shot'],
                        help="Scenario for evaluating the model.")
    parser.add_argument("--model_name", type=str, default='roberta_large_mnli',
                        choices=['roberta_base', 'roberta_large', 'roberta_large_wiki_books', 'roberta_large_mnli'],
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
