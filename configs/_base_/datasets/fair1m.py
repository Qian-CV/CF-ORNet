# dataset settings
dataset_type = 'FAIR1MDataset'
data_root = '/media/ubuntu/CE425F4D425F3983/datasets/FAIR1M2.0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version='le90'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'train/part1/train.txt',
        ann_subdir=data_root + 'train/part1/labelXml/',
        img_subdir=data_root + 'train/part1/images/',
        pipeline=train_pipeline,
        version='oc'),
    val=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'validation/validation.txt',
        ann_subdir=data_root + 'validation/labelXmls/labelXml/',
        img_subdir=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='oc'),
    test=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'validation/validation.txt',
        ann_subdir=data_root + 'validation/labelXmls/labelXml/',
        img_subdir=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='oc')
)
