export PYTHONPATH=../NSP-BERT/
for i in 1 2 3 4 5
do
  python ./baselines/efl_classification_roberta.py \
  --method few-shot \
  --n_th_set $i \
  --device 0 \
  --with_nli_pretrain \
  --dataset_name SST-2 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --model_name roberta_large_mnli
done