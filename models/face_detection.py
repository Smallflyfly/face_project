#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/28
"""
import torch
from torch.backends import cudnn

from data import cfg_re50
from models.retinaface import RetinaFace
from utils.net_utils import load_model, image_process, process_face_data

cfg = cfg_re50
retina_trained_model = './weights/Resnet50_Final.pth'
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net = load_model(retina_net, retina_trained_model, False)
retina_net = retina_net.cuda(0)
retina_net.eval()
cudnn.benchmark = True
device = torch.device("cuda")


def face_detection(im):
    resize = 1
    im, im_width, im_height, scale = image_process(im, device)
    loc, conf, landms = retina_net(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data