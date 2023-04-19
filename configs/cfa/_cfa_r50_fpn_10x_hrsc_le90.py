_base_ = ['../rotated_reppoints/_rotated_reppoints_r50_fpn_10x_hrsc_le90.py']

model = dict(
    bbox_head=dict(use_reassign=True),
    train_cfg=dict(
        refine=dict(assigner=dict(pos_iou_thr=0.1, neg_iou_thr=0.1))))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
work_dir = './work_dirs/cfa/cfa_r50_fpn_10x_hrsc_le90'
auto_resume = False
gpu_ids = range(0, 1)
evaluation = dict(interval=12, metric='mAP')