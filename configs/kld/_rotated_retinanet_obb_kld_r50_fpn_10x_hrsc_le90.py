_base_ = ['../rotated_retinanet/_rotated_retinanet_obb_r50_fpn_10x_hrsc_le90.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            _delete_=True,
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1,
            loss_weight=10.0)))

data = dict(samples_per_gpu=4, workers_per_gpu=2)
work_dir = './work_dirs/kld/rotated_retinanet_obb_kld_r50_fpn_10x_hrsc_le90'
auto_resume = False
gpu_ids = range(0, 1)
evaluation = dict(interval=12, metric='mAP')
