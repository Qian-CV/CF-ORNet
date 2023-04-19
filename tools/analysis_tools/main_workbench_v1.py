import argparse
import xml.etree.cElementTree as ET
from mmrotate.core.bbox.transforms import obb2poly, obb2poly_le90
import torch
import numpy as np
import os
import glob
import mmcv
from tqdm import tqdm
from mmrotate.datasets.hrsc import HRSCDataset
from DOTA_devkit import hrsc2016fg_evaluation as hrsc_eva
from classification import cls_on_merged_txt as cls
import time
from NMS_tool.rnms_python import rnms


def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--convert-mode', help='选择转化模式：1——转换数据集；'
                                             '2——转换检测结果为多边形标注; 3——显示检测评估的mAP;'
                                             ' 4——进行对检测结果的重分类、NMS、评估mAp')
    parser.add_argument('--res-dir', help='选择包含pkl检测结果文件的目录')
    parser.add_argument('--thresh', help='设定输入的重分类阈值')
    parser.add_argument('--data-type', help='选择HRSC数据集或VEDAI数据集')
    args = parser.parse_args()
    return args


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    img_id = None
    box_list = []
    for child_of_root in root:
        if child_of_root.tag == 'Img_ID':
            img_id = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeWidth':
            img_width = int(child_of_root.text)
        if child_of_root.tag == 'Img_SizeHeight':
            img_height = int(child_of_root.text)
        if child_of_root.tag == 'HRSC_Objects':
            box_list = []
            for child_item in child_of_root:
                if child_item.tag == 'HRSC_Object':
                    label = 1
                    # for child_object in child_item:
                    #     if child_object.tag == 'Class_ID':
                    #         label = NAME_LABEL_MAP[child_object.text]
                    tmp_box = [0., 0., 0., 0., 0.]
                    for node in child_item:
                        if node.tag == 'mbox_cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'mbox_cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'mbox_w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'mbox_h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'mbox_ang':
                            tmp_box[4] = float(node.text)

                    tmp_box = obb2poly_le90(torch.Tensor(tmp_box).unsqueeze(0))
                    # assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    # if len(tmp_box) != 0:
                    box_list.append(tmp_box)
            # box_list = coordinate_convert(box_list)
            # print(box_list)
    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_id, img_height, img_width, gtbox_label


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def res_concert(testset_img_path, result_file_root, version):
    """
    Args:
        testset_img_path:测试集图像测试set文件的目录
        result_file_root: 检测网络所得到的.pkl文件所在目录
    Returns:
    """
    result_file = glob.glob(result_file_root + '/*.pkl')
    det_results = np.array(mmcv.load(result_file[0]))
    save_dir = result_file_root + '/wait4cla/'
    mkdir(save_dir)
    classes = HRSCDataset.HRSC_CLASSES
    # 读取400+图片的编号
    f_read = open(testset_img_path, 'r')
    imgs_id = []
    lines = f_read.readlines()
    for line in lines:
        line = line.strip('\n')
        imgs_id.append(line)
    # print(imgs_id)
    # 创建22个txt文档
    for cls in classes:
        f = open(os.path.join(save_dir, cls + ".txt"), 'w')
        # 写入txt中
        for pic_num, picture in enumerate(det_results):
            for class_num, bboxes in enumerate(picture):
                if classes[class_num] != cls:
                    continue
                if bboxes.size != 0:
                    for bbox in bboxes:
                    # 计算转换后的八点坐标
                        score = bbox[5]
                        bbox = obb2poly(torch.Tensor(bbox[:5]).unsqueeze(0), version=version)
                        # print(bbox)
                        bbox = bbox.squeeze(0).tolist()
                        # print(bbox, type(bbox))
                        f.write(f'{imgs_id[pic_num]} {score:.4f} {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} {bbox[4]:.4f} {bbox[5]:.4f} {bbox[6]:.4f} {bbox[7]:.4f}\n')
        f.close()


def res_evaluate(src_txt_path, gt_dir, imagesetfile):
    """

    Args:
        src_txt_path: 原始txt文本的路径
        gt_dir: 测试的标注文件路径
        imagesetfile: 测试集图像测试set文件的路径

    Returns:

    """
    detpath = hrsc_eva.osp.join(src_txt_path, '{:s}.txt')
    # print(detpath)
    annopath = hrsc_eva.osp.join(gt_dir, '{:s}.xml')
    # print(annopath)

    # For HRSC2016
    # 不包含01ship和03warcraft
    classnames = ['100000005',
                  '100000006',
                  '100000007',
                  '100000008',
                  '100000009',
                  '100000010',
                  '100000011',
                  '100000013',
                  '100000015',
                  '100000016',
                  '100000018',
                  '100000019',
                  '100000020',
                  '100000022',
                  '100000024',
                  '100000025',
                  '100000026',
                  '100000027',
                  '100000028',
                  '100000029',
                  '100000030',
                  '100000032']
    classaps = []
    map = 0
    for classname in classnames:
        # print('classname:', classname)
        rec, prec, ap = hrsc_eva.voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 classnames,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print(classname, ":", ap)
        classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
    # plt.show()
    map = map / len(classnames) *100
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)
    return map, classaps.tolist()


def nms_txt(txt_dir):
    mkdir(os.path.join(txt_dir, 'result_after_FgAndNms'))
    '''
    对每个类别，依次进行NMS
    '''

    for txt in os.listdir(txt_dir):  # 依次进行各个类别的NMS
        clsname_txt = txt
        print(txt)
        results_all_imgs = {}
        results_after_nms = []  # 每个类别NMS后存放结果的列表
        results = []
        if os.path.splitext(txt)[1] != '.txt':
            continue
        txt = os.path.join(txt_dir, txt)
        with open(txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                result = line.strip().split()
                results.append(result)
        for result in results:
            if result[0] not in results_all_imgs.keys():
                results_all_imgs[result[0]] = []
            results_all_imgs[result[0]].append(result)

        for key in results_all_imgs.keys():
            results_one_img = []
            results_one_img.append(results_all_imgs[key])
            results_one_img = torch.Tensor(np.array(results_one_img).astype(float)).squeeze_()
            try:
                boxes = results_one_img[:, 2:]
                scores = results_one_img[:, 1]
            except:
                boxes = results_one_img[2:].unsqueeze_(0)
                scores = results_one_img[1].unsqueeze_(0)
            boxes_per_cls_per_img = np.array(rnms(boxes, scores, score_thresh=0, nms_thresh=0.1)).tolist()
            for box in boxes_per_cls_per_img:
                box_ = ' '.join(str(x) for x in box)
                results_after_nms.append(key.split('.txt')[0] + ' ' + box_)
        with open(os.path.join(txt_dir, 'result_after_FgAndNms', clsname_txt), 'w') as f1:
            for line in results_after_nms:
                f1.write(line + '\n')
            f1.close()
    print('非极大值抑制完成！')


if __name__ == '__main__':
    # 选择转化模式：1——转换数据集；2——转换检测结果为多边形标注; 3——显示检测评估的mAP; 4——进行对检测结果的重分类、NMS、评估mAp
    # 设置模式：
    args = parse_args()
    assert args.res_dir or args.convert_mode, \
        ('Please input the argument "--res-dir" and "--convert-mode"'
         )
    # 唯一需要改得路径：
    # result_file_root = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/sasm/sasm_reppoints_r50_fpn_10x_hrsc_oc/'
    # convert_mode = '4'
    result_file_root = args.res_dir
    convert_mode = args.convert_mode
    data_type = args.data_type

    # 判断是什么数据集，从而选择对应路径
    img_dir = '/media/ubuntu/nvidia/HRSC2016fg/test/AllImages/'
    gt_dir = '/media/ubuntu/nvidia/HRSC2016fg/test/Annotations/'
    imagesetfile = '/media/ubuntu/nvidia/HRSC2016fg/test/test.txt'


    if convert_mode == '1':
        data_root = 'H:/数据集/remote_sensing_fine_grained/HRSC2016fg/'
        src_xml_path = data_root + 'test/Annotations'
        txt_path = data_root + 'test/labelTxt'
        mkdir(txt_path)
        print('正在加载源数据')
        src_xmls = glob.glob(f'{src_xml_path}/*.xml')
        print('开始转换')
        for src_xml in tqdm(src_xmls):
            try:
                # ori_image = cv.imread(img_path)
                # x_path = img_path[:-3].replace('AllImages', 'Annotations') + 'xml'
                img_id, img_height, img_width, gtbox_labels = read_xml_gtbox_and_label(src_xml)
                if len(gtbox_labels) == 0:
                    continue
                for ann in gtbox_labels:
                    with open(f"{txt_path}/{img_id}.txt", 'a+') as f:
                        f.write(f"{ann[0]:.4f} {ann[1]:.4f} {ann[2]:.4f} {ann[3]:.4f} {ann[4]:.4f} {ann[5]:.4f} {ann[6]:.4f} {ann[7]:.4f} ship 0\n")

            except:
                print('Error', src_xml_path)

    if convert_mode == '2':
        res_concert(imagesetfile, result_file_root, version='le90')
        print('由旋转框==》多边形框，转换完成！')

    if convert_mode == '3':
        src_txt_path = result_file_root + '/wait4cla/'
        map, classaps = res_evaluate(src_txt_path, gt_dir, imagesetfile)
        f = open(os.path.join(src_txt_path, '0评估结果记录.log'), 'w')
        f.write(f'mAP: {map:.2f} \nAPs: {classaps} \n')
        f.close()

    if convert_mode == '4':
        # 进行二次分类并产生结果
        src_txt_path = result_file_root + '/wait4cla/'
        # resnet18权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp5_2/res18_middle_99.4813_mmrotate.pth'
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp5_2/res18_high.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_paper_ckpt_set_/exp5_2/_resnet18_hrsc2016.py'
        # img_dir = 'VEDAI_1024/test/images/'
        # resnet50权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_secondCls_ckpt_/resnet_epoch_1000.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_secondCls_ckpt_/resnet50_hrsc2016.py'
        # swin权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_secondCls_ckpt_/swin_epoch_98.85892.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_secondCls_ckpt_/_swin-tiny_16xb64_in500_hrsc2016.py'
        # renest50权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_secondCls_ckpt_/resnest50_epoch_99.688.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/1mmrotate_main/tools/work_dirs/_secondCls_ckpt_/_resnest50_hrsc2016.py'
        # resnet101权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/resnet101_hrsc2016/epoch_99.38.pth'
        checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/resnet101_vedai/epoch_200.pth'
        config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/resnet101_hrsc2016/_resnet101_hrsc2016.py'
        # resnest101权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/resnest101_hrsc2016/epoch_99.17.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/resnest101_hrsc2016/_resnest101_hrsc2016.py'
        # convNext 权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/convnext_hrsc2016/epoch_99.689.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/convnext_hrsc2016/_convnext-xlarge_64xb64_hrsc2016.py'
        # Deit权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/deit_hrsc2016/epoch_99.17.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/deit_hrsc2016/_deit-base-distilled_ft-16xb32_hrsc2016-384px.py'
        # vit权重
        # checkpoint = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/vit_224_hrsc2016/epoch_95.54.pth'
        # config_dir = '/media/ubuntu/nvidia/wlq/CTF_ORnet/3mmclassification/tools/work_dirs/vit_224_hrsc2016/_vit-base-p16_pt-64xb64_hrsc2016-224.py'
        if len(args.thresh.split()) == 1:
            thresh = float(args.thresh)
            tic = time.time()
            TOTAL_TIME = cls.read_txts(src_txt_path, img_dir, checkpoint, config_dir, thresh=thresh)
            toc = time.time()
            print('TOTAL_TIME', TOTAL_TIME)
            print('用时：', toc - tic)  # 用时
            print('二次分类完成！')
            # 进行非极大值抑制
            txt_dir = src_txt_path + 'result_after_finegrain/'
            nms_txt(txt_dir)
            # 进行结果评估
            txt_dir = txt_dir + 'result_after_FgAndNms/'
            map, classaps = res_evaluate(txt_dir, gt_dir, imagesetfile)
            f = open(os.path.join(src_txt_path, '0评估结果记录.log'), 'r+')
            old_map = float(f.readlines()[0].split(' ')[1])
            diff = map - old_map
            f.write(f'二次分类(thresh={thresh})+NMS后：\nmAP: {map:.2f},变化了{diff:.2f}\nAPs: {classaps}\n')
            f.close()
        else:
            thresh_list = args.thresh.split()
            threshs = list(np.arange(float(thresh_list[0]), float(thresh_list[1]),
                                     float(thresh_list[2])))
            threshs.append(float(thresh_list[1]))
            for thresh in threshs:
                thresh_float = float(thresh)
                print(f'开始处理{thresh}权重')
                tic = time.time()
                TOTAL_TIME = cls.read_txts(src_txt_path, img_dir, checkpoint, config_dir, thresh=thresh_float)
                toc = time.time()
                print('TOTAL_TIME', TOTAL_TIME)
                print('用时：', toc - tic)  # 用时
                print('二次分类完成！')
                # 进行非极大值抑制
                txt_dir = src_txt_path + 'result_after_finegrain/'
                nms_txt(txt_dir)
                # 进行结果评估
                txt_dir = txt_dir + 'result_after_FgAndNms/'
                map, classaps = res_evaluate(txt_dir, gt_dir, imagesetfile)
                f = open(os.path.join(src_txt_path, '0评估结果记录.log'), 'r+')
                old_map = float(f.readlines()[0].split(' ')[1])
                diff = map - old_map
                f.write(f'二次分类(thresh={thresh})+NMS后：\nmAP: {map:.2f},变化了{diff:.2f}\nAPs: {classaps}\n')
                f.close()




