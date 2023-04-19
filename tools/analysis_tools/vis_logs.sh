#!bin/bash
python ./analyze_logs.py \
plot_curve /media/ubuntu/nvidia/wlq/2all_works/{新}mmrotate与mmrazor/mmrazor1/tools/mmrotate/work_dirs/5wlq_cwd_cls_head_s2anet_r50_fpn_s2anet_r50_fpn_10x_hrsc2016/20220528_003450.log.json \
	--keys mAP
#python ./analyze_logs.py cal_train_time work_dirs/wlq_test/20220422_095303.log.json