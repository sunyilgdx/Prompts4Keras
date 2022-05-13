export PYTHONPATH=../Prompts4Keras/
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