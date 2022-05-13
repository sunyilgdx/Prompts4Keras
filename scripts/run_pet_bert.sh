export PYTHONPATH=../NSP-BERT/
for i in 1 2 3 4 5
do
  python ./baselines/pet_classification_bert.py \
  --method few-shot \
  --n_th_set $i \
  --device 0 \
  --dataset_name SST-2 \
  --batch_size 8 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --model_name bert_large
done