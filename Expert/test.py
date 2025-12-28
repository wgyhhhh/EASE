import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.EASE import EASE
import json
from engine import *
import os
import sys
import numpy
from utils import *
from configs.config_setting import setting_config
import tempfile
import warnings
import argparse
from engine import calculate_metrics

warnings.filterwarnings("ignore")
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

def main():
    print('#----------Parse all arguments----------#')
    args = parse_args()
    config = setting_config
    config.seed = args.seed
    config.gpu_id = args.gpu_id
    config.bert_path = args.bert_path
    config.datasets = args.datasets
    config.work_dir = f'./results/'

    if config.datasets == 'weibo':
        config.data_path = './data/weibo'
    elif config.datasets == 'weibo21':
        config.data_path = './data/weibo21/'
    elif config.datasets == 'gossipcop':
        config.data_path = './data/gossipcop/'
    

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    from datasets.dataloader import get_dataloader

    test_path = os.path.join(config.data_path, 'test.json')

    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentiment_data, reasoning_data, evidence_data = [], [], []

    for item in data:
        evidence_reliable = item.get('evidence_reliable', 1)
        reasoning_reliable = item.get('reasoning_reliable', 0)
        
        if evidence_reliable == 1:
            evidence_data.append(item)
        elif evidence_reliable == 0 and reasoning_reliable == 1:
            reasoning_data.append(item)
        else:
            sentiment_data.append(item)

    def create_loader(subset_data):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
            json.dump(subset_data, tmp, ensure_ascii=False)
            tmp_path = tmp.name
        
        loader = get_dataloader(tmp_path, config.max_len, 1, False, config.bert_path)
        os.unlink(tmp_path)  
        return loader

    test_loader_sentiment = create_loader(sentiment_data) if sentiment_data else None
    test_loader_reasoning = create_loader(reasoning_data) if reasoning_data else None
    test_loader_evidence = create_loader(evidence_data) if evidence_data else None

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    sentiment_expert = EASE(
            config,
            expert_name = 'sentiment',
            emb_dim=model_cfg['emb_dim'],
            co_attention_dim=model_cfg['co_attention_dim'],
            mlp_dims = model_cfg['mlp_dims'],
            mlp_dropout = model_cfg['mlp_dropout']
        )
    reasoning_expert = EASE(
            config,
            expert_name = 'reasoning',
            emb_dim=model_cfg['emb_dim'],
            co_attention_dim=model_cfg['co_attention_dim'],
            mlp_dims = model_cfg['mlp_dims'],
            mlp_dropout = model_cfg['mlp_dropout']
        )
    evidence_expert = EASE(
            config,
            expert_name = 'evidence',
            emb_dim=model_cfg['emb_dim'],
            co_attention_dim=model_cfg['co_attention_dim'],
            mlp_dims = model_cfg['mlp_dims'],
            mlp_dropout = model_cfg['mlp_dropout']
        )
    
    sentiment_expert = sentiment_expert.cuda()
    reasoning_expert = reasoning_expert.cuda()
    evidence_expert = evidence_expert.cuda()

    print('#----------Testing----------#')
    
    os.makedirs(config.work_dir, exist_ok=True)
        
    logger = get_logger('test', os.path.join(config.work_dir, 'log'))
    
    
    best_weight_sentiment = torch.load(config.work_dir + f'EASE_sentiment_{config.datasets}/checkpoints/parameter_sentiment_{config.datasets}.pkl', map_location=torch.device('cpu'))
    sentiment_expert.load_state_dict(best_weight_sentiment)

    results, labels_sentiment, preds_sentiment, ids = test_one_epoch(
        test_loader=test_loader_evidence,
        model=sentiment_expert,
        logger=logger,   
        config=config,  
        test_data_name=config.datasets  
    )

    
    best_weight_reasoning = torch.load(config.work_dir + f'EASE_reasoning_{config.datasets}/checkpoints/parameter_reasoning_{config.datasets}.pkl', map_location=torch.device('cpu'))
    reasoning_expert.load_state_dict(best_weight_reasoning)
    results, labels_reasoning, preds_reasoning, ids = test_one_epoch(
        test_loader=test_loader_evidence,
        model=reasoning_expert,
        logger=logger,   
        config=config,   
        test_data_name=config.datasets  
    )

    
    best_weight_evidence = torch.load(config.work_dir + f'EASE_evidence_{config.datasets}/checkpoints/parameter_evidence_{config.datasets}.pkl', map_location=torch.device('cpu'))
    evidence_expert.load_state_dict(best_weight_evidence)
    results, labels_evidence, preds_evidence, ids = test_one_epoch(
        test_loader=test_loader_evidence,
        model=evidence_expert,
        logger=logger,   
        config=config,   
        test_data_name=config.datasets  
    )        

    all_labels = []
    all_preds = []
    all_labels.extend(labels_sentiment) 
    all_preds.extend(preds_sentiment)

    all_labels.extend(labels_reasoning)
    all_preds.extend(preds_reasoning)

    all_labels.extend(labels_evidence)
    all_preds.extend(preds_evidence)

    expert_counts = {
    'sentiment': len(labels_sentiment),
    'reasoning': len(labels_reasoning),
    'evidence': len(labels_evidence)
    }

    total_samples = sum(expert_counts.values())

    print("\n" + "="*60)
    print("Expert Usage Statistics:")
    print("="*60)

    # Calculate and print the percentage of calls for each expert
    for expert_name, count in expert_counts.items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {expert_name.capitalize()} Expert: {count} samples ({percentage:.2f}%)")

    print(f"  Total: {total_samples} samples")
    print("="*60)

    final_results = calculate_metrics(all_labels, all_preds)
    
    print(final_results)
    print("="*60 + "\n")
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--seed', type=int, default=3759, help='random seed')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--bert_path', type=str,
                        default='./bert/chinese-bert-wwm-ext',
                        help='BERT model path')
    parser.add_argument('--datasets', type=str, default='weibo21', help='dataset name')
    return parser.parse_args()


if __name__ == '__main__':
    config = setting_config
    main()