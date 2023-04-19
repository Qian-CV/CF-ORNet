_base_ = ['../rotated_retinanet/_rotated_retinanet_obb_r50_fpn_10x_hrsc_le90.py']

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='KFIoURRetinaHead',
        num_classes=22,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='KFLoss', loss_weight=5.0)))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
work_dir = './work_dirs/kfiou/rotated_retinanet_obb_kfiou_r50_fpn_10x_hrsc_le90'
auto_resume = False
gpu_ids = range(0, 1)
evaluation = dict(interval=4, metric='mAP')