import os
from datetime import datetime


class setting_config:
    """
    the config of fake news detection training setting.
    """

    network = 'expert'
    model_config = {
        'expert_name': 'sentiment',
        'emb_dim': 768,
        'co_attention_dim': 300,
        'mlp_dims': [384],
        'mlp_dropout': 0.2,
        'analyzer_parameter': 0.5
    }

    datasets = 'weibo'
    if datasets == 'weibo':
        data_path = './data/weibo'
    elif datasets == 'weibo21':
        data_path = './data/weibo21/'
    elif datasets == 'gossipcop':
        data_path = './data/gossipcop/'
    else:
        raise Exception('datasets in not right!')

    criterion = 'BCELoss'

    num_classes = 1
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 3759
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 1
    max_len = 197
    early_stop = 10
    use_cuda = True
    work_dir = 'results/' + network + '_' + datasets + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 100
    threshold = 0.5

    # Expert configuration
    sentiment_expert = True
    reasoning_expert = True
    evidence_expert = True
    expert_type = 'sentiment'  # or 'reasoning', 'evidence'
    bert_path = './bert/chinese-bert-wwm-ext'
    parameter_pkl_name = 'model.pkl'
    eval_mode = False

    opt = 'Adam'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adam':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False

    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'CosineAnnealingLR':
        T_max = 50
        eta_min = 0.00001
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
