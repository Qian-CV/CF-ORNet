# evaluation
evaluation = dict(interval=8, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[100, 160, 200, 220])
runner = dict(type='EpochBasedRunner', max_epochs=240)
checkpoint_config = dict(interval=2)
