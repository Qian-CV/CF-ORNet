# dataset settings
dataset_type = 'HRSCDataset'
data_root = '/media/ubuntu/CE425F4D425F3983/datasets/HRSC2016fg/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 512)),
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
        img_scale=(800, 512),
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
        classwise=True,
        ann_file=data_root + 'trainval/trainval.txt',
        ann_subdir=data_root + 'trainval/Annotations/',
        img_subdir=data_root + 'trainval/AllImages/',
        pipeline=train_pipeline,
        version='le90'),
    val=dict(
        type=dataset_type,
        classwise=True,
        ann_file=data_root + 'test/test.txt',
        ann_subdir=data_root + 'test/Annotations/',
        img_subdir=data_root + 'test/AllImages/',
        pipeline=test_pipeline,
        version='le90'),
    test=dict(
        type=dataset_type,
        classwise=True,
        ann_file=data_root + 'test/test.txt',
        ann_subdir=data_root + 'test/Annotations/',
        img_subdir=data_root + 'test/AllImages/',
        pipeline=test_pipeline,
        version='le90')
)
