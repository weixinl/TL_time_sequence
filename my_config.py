
ucihar_small_baseline_config = {
    "attr_num":9,
    "window_size":128,
    "label_num": 6,
    "group_num":5,
    "domain_num":4,

    'epoch_num': 100,

    'lr': 0.001,

    'batch_size': 16,

    'momentum': 0.9
}

ucihar_small_transfer_config = {
    "attr_num":9,
    "window_size":128,
    "label_num": 6,
    "group_num":5,
    "domain_num":4,
    "domain_branch_epoch_num":100,
    "label_branch_epoch_num":400,
    'epoch_num': 50,
    "domain_branch_lr":0.001,
    "label_branch_lr":0.0005,
    'lr': 0.0001,
    "domain_branch_batch_size":16,
    "label_branch_batch_size":16,
    'batch_size': 16,
    "domain_branch_momentum":0.9,
    "label_branch_momentum":0.9,
    'momentum': 0.9
}