_base_ = ['../rotated_reppoints/_rotated_reppoints_r50_fpn_10x_hrsc_le90.py']

angle_version = 'le90'
model = dict(
    bbox_head=dict(
        version=angle_version,
        type='KLDRepPointsHead',
        loss_bbox_init=dict(type='KLDRepPointsLoss'),
        loss_bbox_refine=dict(type='KLDRepPointsLoss')),
    train_cfg=dict(
        refine=dict(
            assigner=dict(_delete_=True, type='ATSSKldAssigner', topk=9))))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
work_dir = './work_dirs/g_rep/g_reppoints_r50_fpn_10x_hrsc_le90'
auto_resume = False
gpu_ids = range(0, 1)
evaluation = dict(interval=4, metric='mAP')
