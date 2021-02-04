#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/02/04
"""
import torch

from data import cfg_mnet
from models.retinaface import RetinaFace
from utils.net_utils import load_model

retina_face_model = './weights/mobilenet0.25_Final.pth'
cfg = cfg_mnet
retina_onnx = "retina_face.onnx"


def torch2onnx():
    retina_net = RetinaFace(cfg=cfg, phase='test')
    retina_net = load_model(retina_net, retina_face_model, load_to_cpu=False).cuda()
    retina_net.eval()
    dummy_input = torch.rand(1, 3, 640, 640).cuda()
    try:
        torch.onnx.export(retina_net, dummy_input, retina_onnx, verbose=False, do_constant_folding=False,
                    input_names=['input'], output_names=['output'], opset_version=11)
    except:
        print('onnx 导出失败')
        raise RuntimeError


if __name__ == '__main__':
    torch2onnx()
