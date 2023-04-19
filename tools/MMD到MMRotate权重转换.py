"""
用于将mmd权重转换为mmrotate可以直接用的权重
"""
# TODO: 未完成
import torch
import os

checkpoint = '/root/PycharmProjects/mmrazor/tools/mmcls/work_dirs/wsld_cls_head_resnet50_resnet18_8xb32_in1k_grafting1/epoch_88_9409.pth'

filename = os.path.splitext(checkpoint)[0]
file = torch.load(checkpoint)['state_dict']
transmap = {'bbox_head.fam_reg_:': 'fam_head.reg_convs'}

new_state_dict = {}
for i in file.items():
    if 'distiller' in i[0]:
        continue
    new_state_dict['.'.join(i[0].split('.')[2:])] = i[1]
# for i in new_state_dict.items():
#     print(i)
new_checkpoint = {}
new_checkpoint['meta'] = torch.load(checkpoint)['meta']
new_checkpoint['state_dict'] = new_state_dict
new_checkpoint['optimizer'] = torch.load(checkpoint)['optimizer']
torch.save(new_checkpoint, filename + '_mmc' + '.pth')

s = torch.load(filename + '_mmc' + '.pth')
for i in s['state_dict']:
    print(i)
