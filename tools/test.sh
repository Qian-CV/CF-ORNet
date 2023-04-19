#!/bin/bash
python test.py \
/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp8/CF_ORnet/_oriented_rcnn_r101_fpn_10x_hrsc_le90.py \
/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp8/CF_ORnet/epoch_84.34.pth \
	--eval mAP

#--out /media/ubuntu/nvidia/wlq/2all_works/{新}mmrotate与mmrazor/mmrotate1/tools/work_dirs/wlq_test/result.pkl
#--show

#CUDA_VISIBLE_DEVICES=0 python test.py \
#/root/PycharmProjects/mmrotate/configs/SRep-RDet/SRep-RDet_deploy_10x_hrsc_le135.py \
#/root/PycharmProjects/mmrotate/tools/work_dirs/SRep-RDet_train_10x_hrsc_le135/epoch_110_deploy.pth \
#--out /root/PycharmProjects/mmrotate/tools/work_dirs/SRep-RDet_train_10x_hrsc_le135/res.pkl \
#--eval mAP
