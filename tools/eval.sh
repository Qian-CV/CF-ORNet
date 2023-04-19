#!/bin/bash
python test.py \
/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp8/CF_ORnet/_oriented_rcnn_r101_fpn_10x_hrsc_le90.py \
/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp8/CF_ORnet/epoch_84.34.pth \
	--eval mAP \
	--out /media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp8/CF_ORnet/res.pkl
#
work_dir="/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp8/CF_ORnet"
python ./analysis_tools/main_workbench_v2.py \
	--res-dir $work_dir --data-type 'hrsc' --convert-mode 2
python ./analysis_tools/main_workbench_v2.py \
	--res-dir $work_dir --data-type 'hrsc' --convert-mode 3
python ./analysis_tools/main_workbench_v2.py \
	--res-dir $work_dir --data-type 'hrsc' --convert-mode 4 --thresh '0 1 0.1'
cp $work_dir/wait4cla/0评估结果记录.log $work_dir/blurring_results/
mv $work_dir/blurring_results/0评估结果记录.log $work_dir/blurring_results/"ck84.34_clouds2_orcnn_101评估结果记录(0-1-0.1)resnet101.log"

#work_dir="/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/redet/redet_re50_refpn_10x_vedai_le90"
#python ./analysis_tools/main_workbench_v2.py \
#	--res-dir $work_dir --data-type 'vedai' --convert-mode 2
#python ./analysis_tools/main_workbench_v2.py \
#	--res-dir $work_dir --data-type 'vedai' --convert-mode 3
#python ./analysis_tools/main_workbench_v2.py \
#	--res-dir $work_dir --data-type 'vedai' --convert-mode 4 --thresh '0 1 0.1'
#cp $work_dir/wait4cla/0评估结果记录.log $work_dir
#mv $work_dir/0评估结果记录.log $work_dir/"0ck_redet2评估结果记录(0-1-0.1)resnet101.log"
#
#work_dir="/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/oriented_Rcnn/oriented_rcnn_r101_fpn_10x_vedai_le90"
#python ./analysis_tools/main_workbench_v2.py \
#	--res-dir $work_dir --data-type 'vedai' --convert-mode 2
#python ./analysis_tools/main_workbench_v2.py \
#	--res-dir $work_dir --data-type 'vedai' --convert-mode 3
#python ./analysis_tools/main_workbench_v2.py \
#	--res-dir $work_dir --data-type 'vedai' --convert-mode 4 --thresh '0 1 0.1'
#cp $work_dir/wait4cla/0评估结果记录.log $work_dir
#mv $work_dir/0评估结果记录.log $work_dir/"0ck_ORcnn2评估结果记录(0-1-0.1)resnet101.log"

