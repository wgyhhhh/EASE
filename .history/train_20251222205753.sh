#expert
python train.py \
    --batch_size 64 \
    --epochs 1 \
    --seed 42 \
    --gpu_id "1" \
    --early_stop 5 \
    --bert_path "/home/test3/test3/test3/wgy/fake_news/ARG/bert/chinese-bert-wwm-ext" \
    --expert_type "reasoning" \
    --datasets "weibo21" 