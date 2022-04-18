# dataset settings
dataset_type = 'CustomDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(100, -1), adaptive_side="long"),
    dict(type='Pad', size=(100, 100), pad_val=255, padding_mode="constant"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(100, -1), adaptive_side="long"),
    dict(type='Pad', size=(100, 100), pad_val=255, padding_mode="constant"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix='/home/nojusupov/CASIA-HWDB_Train/Train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/nojusupov/Test',  # TODO: REPLACE WITH REAL VAL
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/home/nojusupov/Test',  # TODO: REPLACE WITH REAL VAL
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
