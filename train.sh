#expert
python train.py \
    --batch_size 64 \
    --epochs 1 \
    --seed 42 \
    --gpu_id "1" \
    --early_stop 15 \
    --bert_path "./bert/chinese-bert-wwm-ext" \
    --expert_type "sentiment" \
    --datasets "weibo"