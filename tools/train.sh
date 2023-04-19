#!/bin/bash
#python train.py ../configs/redet/redet_re50_refpn_10x_hrsc_le90.py
#python train.py ../configs/rotated_reppoints/_rotated_reppoints_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/rotated_reppoints/_rotated_reppoints_r50_fpn_10x_hrsc_oc.py
#python train.py ../configs/gwd/_rotated_retinanet_obb_gwd_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/sasm_reppoints/_sasm_reppoints_r50_fpn_10x_hrsc_le90.py

# 2022/6/18
#python train.py ../configs/kfiou/_s2anet_kfiou_ln_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/kfiou/_r3det_kfiou_ln_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/kfiou/_rotated_retinanet_obb_kfiou_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/g_reppoints/_g_reppoints_r50_fpn_10x_hrsc_le90.py
# 2022/6/18 PM
#python train.py ../configs/sasm_reppoints/_sasm_reppoints_r50_fpn_10x_hrsc_oc.py

# 2022/6/19
#python train.py ../configs/redet/_redet_re18_refpn_10x_hrsc_le90.py
#python train.py ../configs/redet/_redet_re101_refpn_10x_hrsc_le90.py
#python train.py ../configs/redet/_redet_re18_refpn_10x_hrsc_le90_240epoch.py

# 2022/6/22
#python train.py ../configs/oriented_rcnn/oriented_rcnn_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/redet/_redet_re34_refpn_10x_hrsc_le90.py
#python train.py ../configs/redet/_redet_re18_refpn_20x_hrsc_le90.py \
#  --work-dir './work_dirs/redet/redet_re18+re50pre_refpn_20x_hrsc_le90'

# 2022/6/22
#python train.py ../configs/oriented_rcnn/_oriented_rcnn_r18_fpn_10x_hrsc_le90.py
#python train.py ../configs/oriented_rcnn/_oriented_rcnn_r34_fpn_10x_hrsc_le90.py

# 2022/6/26
#work_dir="/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/oriented_Rcnn"
#mkdir $work_dir/oriented_rcnn_r34_fpn_10x_hrsc_le90_grafting1
#mkdir $work_dir/oriented_rcnn_r34_fpn_10x_hrsc_le90_grafting2
#mkdir $work_dir/grafting
#python train.py ../configs/oriented_rcnn/_oriented_rcnn_r34_fpn_10x_hrsc_le90_grafting1.py \
#>./work_dirs/oriented_Rcnn/oriented_rcnn_r34_fpn_10x_hrsc_le90_grafting1/1.out &
#
#python train.py ../configs/oriented_rcnn/_oriented_rcnn_r34_fpn_10x_hrsc_le90_grafting2.py \
#>./work_dirs/oriented_Rcnn/oriented_rcnn_r34_fpn_10x_hrsc_le90_grafting2/2.out &

# 2022/6/30
#python train.py ../configs/oriented_rcnn/_oriented_rcnn_r101_fpn_10x_hrsc_le90.py \
#  --work-dir ./work_dirs/oriented_Rcnn/_oriented_rcnn_r101_fpn_10x_hrsc_le90_2

# 2022/7/6
#python train.py ../configs/s2anet/s2anet_r50_fpn_10x_VEDAI.py
#python train.py ../configs/sasm_reppoints/_sasm_reppoints_r50_fpn_10x_vedai_oc.py
#python train.py ../configs/redet/_redet_re50_refpn_10x_vedai_le90.py
#python train.py ../configs/oriented_rcnn/_oriented_rcnn_r50_fpn_10x_vedai_le90.py

# 2022/7/11
#python train.py ../configs/r3det/_r3det_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/rotated_fcos/_rotated_fcos_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/rotated_reppoints/_rotated_reppoints_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/rotated_retinanet/_rotated_retinanet_obb_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/rotated_faster_rcnn/_rotated_faster_rcnn_r50_fpn_10x_hrsc_le90.py
#python train.py ../configs/roi_trans/_roi_trans_r50_fpn_10x_hrsc_le90.py

# 2022/7/12
python train.py ../configs/oriented_rcnn/_oriented_rcnn_r101_fpn_10x_vedai_le90.py