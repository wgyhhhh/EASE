#expert
python train.py \
    --batch_size 64 \
    --epochs 100 \
    --seed 3759 \
    --lr 2e-5 \
    --gpu_id "1" \
    --early_stop 10 \
    --bert_path "/home/test3/test3/test3/wgy/fake_news/ARG/bert/chinese-bert-wwm-ext" \
    --expert_type "evidence" \
    --datasets "weibo21" \
    --analyzer_parameter 1.9