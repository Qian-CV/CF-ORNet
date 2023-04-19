_base_ = ['../rotated_retinanet/_rotated_retinanet_obb_r50_fpn_10x_hrsc_le90.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
work_dir = './work_dirs/gwd/rotated_retinanet_obb_gwd_r50_fpn_10x_hrsc_le90'
auto_resume = False
gpu_ids = range(0, 1)
evaluation = dict(interval=12, metric='mAP')
