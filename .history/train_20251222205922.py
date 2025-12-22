import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.EASE import EASE

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
import argparse

warnings.filterwarnings("ignore")


def main():
    print('#----------Parse all arguments----------#')
    args = parse_args()
    config = setting_config
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.seed = args.seed
    config.gpu_id = args.gpu_id
    config.early_stop = args.early_stop
    config.bert_path = args.bert_path
    config.expert_type = args.expert_type
    config.datasets = args.datasets
    config.work_dir = f'./results/EASE_{config.expert_type}_{config.datasets}/'

    if config.datasets == 'weibo':
        config.data_path = './data/weibo'
    elif config.datasets == 'weibo21':
        config.data_path = './data/weibo21/'
    elif config.datasets == 'gossipcop':
        config.data_path = './data/gossipcop/'
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    from datasets.dataloader import get_dataloader

    train_path = os.path.join(config.data_path, 'train.json')
    train_loader = get_dataloader(
        train_path,
        config.max_len,
        config.batch_size,
        shuffle=True,
        bert_path=config.bert_path,
    )

    val_path = os.path.join(config.data_path, 'val.json')
    val_loader = get_dataloader(
        val_path,
        config.max_len,
        config.batch_size,
        shuffle=False,
        bert_path=config.bert_path,
    )

    test_path = os.path.join(config.data_path, 'test.json')
    test_loader = get_dataloader(
        test_path,
        config.max_len,
        config.batch_size,
        shuffle=False,
        bert_path=config.bert_path,
    )

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    model = EASE(
            config,
            expert_name = config.expert_type,
            emb_dim=model_cfg['emb_dim'],
            co_attention_dim=model_cfg['co_attention_dim'],
            mlp_dims = model_cfg['mlp_dims'],
            mlp_dropout = model_cfg['mlp_dropout']
        )
    model = model.cuda()

    cal_params_flops(model, config, logger)

    print('#----------Preparing loss, opt, sch----------#')
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    recorder = Recorder(config.early_stop)
    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        print(f'#---------- Epoch {epoch} ----------#')
        step = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        
        print('#----------Validation----------#')
        val_results = val_one_epoch(
            val_loader,
            model,
            epoch,
            logger,
            config
        )

        mark = recorder.add(val_results)
        if mark == 'save':
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'parameter_{config.expert_type}_{config.datasets}.pkl'))
        if mark == 'esc':
            break

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': val_results['loss'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'latest.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + f'checkpoints/parameter_{config.expert_type}_{config.datasets}.pkl', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        results, labels, preds, ids = test_one_epoch(
            test_loader,
            model,
            logger,
            config
        )

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Config')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--seed', type=int, default=3759, help='random seed')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop patience')
    parser.add_argument('--bert_path', type=str,
                        default='./bert/chinese-bert-wwm-ext',
                        help='BERT model path')
    parser.add_argument('--expert_type', type=str, default='sentiment', help='expert type')
    parser.add_argument('--datasets', type=str, default='weibo', help='dataset name')
    return parser.parse_args()


if __name__ == '__main__':
    config = setting_config
    main()
