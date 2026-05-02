# config.py

cfg_BV4 = {
    'name': 'BBLiteV4',
    'condition_we':[1,1.4,1.9]
}

cfg_MV1 = {
    'name': 'mobilenet0.25',
    'condition_we':[1,1.5,2.1]
}

cfg_SV2 = {
    'name': 'shufflenet_v2_x0_5',
    'condition_we':[1,1.4,1.9]
}

cfg_CAFACLite = {
    'min_sizes': [[16, 24, 32], [64, 96, 128], [256, 384, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 4,
    'epoch': 140,
    'decay1': 100,
    'decay2': 120,
    'image_size': 1024,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 256,
    'out_channel': 32
}

