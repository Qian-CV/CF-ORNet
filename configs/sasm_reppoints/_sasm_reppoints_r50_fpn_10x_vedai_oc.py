_base_ = ['../rotated_reppoints/_rotated_reppoints_r50_fpn_10x_vedai_oc.py']

model = dict(
    bbox_head=dict(
        type='SAMRepPointsHead',
        loss_bbox_init=dict(type='BCConvexGIoULoss', loss_weight=0.375)),

    # training and testing settings
    train_cfg=dict(
        refine=dict(
            _delete_=True,
            assigner=dict(type='SASAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
work_dir = './work_dirs/sasm/sasm_reppoints_r50_fpn_10x_vedai_oc'
auto_resume = False
gpu_ids = range(0, 1)
evaluation = dict(interval=2, metric='mAP')
checkpoint_config = dict(interval=2)
