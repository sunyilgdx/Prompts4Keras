# Prompts4Keras
Prompt-learning methods used BERT4Keras (PET, EFL and NSP-BERT), both for Chinese and English.

## Overview
In order to better compare NSP-BERT and other two basic prompt-learning methods based on MLM and NLI in Chinese and English two languages, and can easily conduct experiments on the BERT4Keras framework, especially transfer the original English RoBERTa model to the BERT4Keras framework, we developed this repository.

### Target
Mainly for text classification tasks in zero-shot and few-shot learning scenarios.

### Supported Methods
- [Standard fine-tuning](https://arxiv.org/pdf/1810.04805.pdf)
- [PET](https://arxiv.org/pdf/2001.07676.pdf)
- [EFL](https://arxiv.org/pdf/2104.14690.pdf)
- [NSP-BERT](https://arxiv.org/abs/2109.03564)

### Supported Models
- BERT for both English and Chinese, and BERT-like Chinese RoBERTa, such as [vanilla BERT](https://github.com/google-research/bert), [HFL Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm), [UER Chinese-BERT](https://github.com/dbiir/UER-py).

- English-RoBERTa proposed by [Fairseq](https://github.com/pytorch/fairseq/tree/main/examples/roberta).

*NOTE: We need to use some scripts to convert the pytorch model to the tensorflow model we used.*

## Environments
Different from the baselines, this repository all uses the BERT4Keras framework, which is completely based on tensorflow.

Since it needs to run on a graphics card of the Ampere framework (such as A100, RTX 3090), we need to install the **NVIDIA version of tensorflow**.
```
bert4keras==0.10.8
fairseq==0.10.2
keras==2.6.0
nvidia_tensorflow==1.15.4+nv20.11
scikit_learn==1.0.2
scipy==1.3.1
torch==1.7.0
transformers==4.12.3
```

## Reproduce experiments
1. Downloading the models
- For all English tasks we use [vanilla BERT-Large, cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)

- For all the Chinese tasks we use [UER-BERT-Base (MixedCorpus+BertEncoder(base)+BertTarget)](https://share.weiyun.com/5QOzPqq)

- For PET, we can choice [English-RoBERTa-Large](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz) proposed by Fairseq, and [English-RoBERTa-Large wiki+books](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.bookswiki.100k.tar.gz)

- For EFL, we need to use model trained on NLI dataset, such as [English RoBERTa-Large-MNLI](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz), or [Chinese-BERT-base-OCNLI]() (we trained by ourselves)

2. Convert pytorch models to tf

- `convert_fairseq_roberta_to_tf.py`
- `convert_bert_from_uer_to_original_tf.py`

3. Using `run_nsp_bert.sh` and other scripts to reproduce our experiments. For each few-shot learning task, we divide the training set and dev set according to 5 random seeds, and conduct experiments separately.
- English tasks
**dataset_name**: `SST-2`, `MR`, `CR`, `Subj`, `MPQA`, `Yahoo!`, `AGNews`.
```
for i in 1 2 3 4 5
do
  python ./nsp_bert/nsp_classification.py \
  --method few-shot \
  --n_th_set $i \
  --device 0 \
  --dataset_name SST-2 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --loss_function BCE \
  --model_name bert_large
done
```

- Chinese tasks
**dataset_name**: `EPRSTMT`, `TNEWS`, `CSLDCP`, `IFLYTEK`.
```
for i in 1 2 3 4 5
do
  python ./nsp_bert/nsp_classification.py \
  --method few-shot \
  --n_th_set $i \
  --device 0 \
  --dataset_name EPRSTMT \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --loss_function BCE \
  --model_name chinese_bert_base
done

