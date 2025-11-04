import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []
    avg_loss_classify = Averager()

    for iter, batch in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()

        # Move data to GPU
        batch_data = data2gpu(batch, config.use_cuda)
        label = batch_data['label']

        # Forward pass
        batch_input_data = {**config.__dict__, **batch_data}
        res = model(**batch_input_data)

        # Calculate losses
        loss_classify = criterion(res['classify_pred'], label.float())
        loss = loss_classify

        # Add auxiliary losses for experts
        if config.expert_type == 'sentiment':
            simple_ftr_2_label = batch_data['FTR_2_pred']
            loss_simple_aux = torch.nn.CrossEntropyLoss()(res['simple_ftr_2_pred'], simple_ftr_2_label.long())
            loss += config.model_config['analyzer_parameter'] * loss_simple_aux
        elif config.expert_type == 'reasoning':
            simple_ftr_3_label = batch_data['FTR_3_pred']
            loss_simple_aux = torch.nn.CrossEntropyLoss()(res['simple_ftr_3_pred'], simple_ftr_3_label.long())
            loss += config.model_config['analyzer_parameter'] * loss_simple_aux
        elif config.expert_type == 'evidence':
            simple_ftr_4_label = batch_data['FTR_4_pred']
            loss_simple_aux = torch.nn.CrossEntropyLoss()(res['simple_ftr_4_pred'], simple_ftr_4_label.long())
            loss += config.model_config['analyzer_parameter'] * loss_simple_aux

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        avg_loss_classify.add(loss_classify.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    scheduler.step()
    return step


def val_one_epoch(val_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    labels = []
    loss_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch_data = data2gpu(batch, config.use_cuda)
            batch_label = batch_data['label']

            batch_input_data = {**config.__dict__, **batch_data}
            res = model(**batch_input_data)

            loss_classify = criterion(res['classify_pred'], batch_label.float())

            labels.extend(batch_label.detach().cpu().numpy().tolist())
            preds.extend(res['classify_pred'].detach().cpu().numpy().tolist())
            loss_list.append(loss_classify.item())

    # Calculate metrics
    results = calculate_metrics(labels, preds, config.threshold)
    results['loss'] = np.mean(loss_list)

    # 总是打印完整指标，移除条件判断
    log_info = (f'val epoch: {epoch}, loss: {results["loss"]:.4f}, '
                f'accuracy: {results["accuracy"]:.4f}, '
                f'precision: {results["precision"]:.4f}, '
                f'recall: {results["recall"]:.4f}, '
                f'f1: {results["f1"]:.4f}, '
                f'auc: {results["auc"]:.4f}')
    print(log_info)
    logger.info(log_info)

    return results

def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    labels = []
    ids = []
    loss_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch_data = data2gpu(batch, config.use_cuda)
            batch_label = batch_data['label']
            batch_id = batch_data.get('id', [])

            batch_input_data = {**config.__dict__, **batch_data}
            res = model(**batch_input_data)

            loss_classify = criterion(res['classify_pred'], batch_label.float())

            labels.extend(batch_label.detach().cpu().numpy().tolist())
            preds.extend(res['classify_pred'].detach().cpu().numpy().tolist())
            ids.extend(batch_id)
            loss_list.append(loss_classify.item())

    # Calculate metrics
    results = calculate_metrics(labels, preds, config.threshold)
    results['loss'] = np.mean(loss_list)

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    log_info = f'test of best model, loss: {results["loss"]:.4f}, accuracy: {results["accuracy"]:.4f}, ' \
               f'precision: {results["precision"]:.4f}, recall: {results["recall"]:.4f}, ' \
               f'f1: {results["f1"]:.4f}, auc: {results["auc"]:.4f}'
    print(log_info)
    logger.info(log_info)

    return results, labels, preds, ids


def calculate_metrics(labels, preds, threshold=0.5):
    """Calculate classification metrics"""
    labels = np.array(labels)
    preds = np.array(preds)

    # Binary predictions
    binary_preds = (preds >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds, zero_division=0)
    recall = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)

    # AUC
    try:
        auc = roc_auc_score(labels, preds)
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


class Averager:
    """Simple average calculator"""

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def data2gpu(batch, use_cuda):
    """Move batch data to GPU"""
    if use_cuda:
        return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    else:
        return batch
