export PYTHONPATH=../Prompts4Keras/
for TASK in SST-2 MR CR MPQA Subj Yahoo AGNews EPRSTMT TNEWS TNEWSK CSLDCP IFLYTEK
    do
    case $TASK in
        SST-2)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        MR)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        CR)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        Subj)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        MPQA)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        Yahoo)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        AGNews)
            BS=8
            LR=2e-5
            LF="BCE"
            MN="bert_large"
            ;;
        EPRSTMT)
            BS=8
            LR=1e-5
            LF="BCE"
            MN="chinese_bert_base"
            ;;
        TNEWS)
            BS=8
            LR=1e-5
            LF="BCE"
            MN="chinese_bert_base"
            ;;
        TNEWSK)
            BS=8
            LR=1e-5
            LF="BCE"
            MN="chinese_bert_base"
            ;;
        CSLDCP)
            BS=1
            LR=1e-5
            LF="BCE"
            MN="chinese_bert_base"
            ;;
        IFLYTEK)
            BS=1
            LR=1e-5
            LF="BCE"
            MN="chinese_bert_base"
            ;;
    esac
    for i in 1 2 3 4 5
    do
      python ./nsp_bert/nsp_classification.py \
      --method few-shot \
      --n_th_set $i \
      --device 0 \
      --epochs 10 \
      --dataset_name $TASK \
      --batch_size $BS \
      --learning_rate $LR \
      --loss_function $LF \
      --model_name $MN
    done

done