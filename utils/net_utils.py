# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 16:05
# @Author  : Fangpf
# @FileName: net_utils.py
import cv2
import torch
import numpy as np
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms,
                      resize, top_k=5000, nms_threshold=0.4, keep_top_k=750):
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.cuda()
    priors_data = priors.data
    boxes = decode(loc.data.squeeze(0), priors_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).cpu().detach().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), priors_data, cfg['variance'])
    scale_landm = torch.from_numpy(np.array([
        im.shape[3], im.shape[2], im.shape[3], im.shape[2],
        im.shape[3], im.shape[2], im.shape[3], im.shape[2],
        im.shape[3], im.shape[2]
    ]))
    scale_landm = scale_landm.float()
    scale_landm = scale_landm.cuda()
    landms = landms * scale_landm / resize
    landms = landms.cpu().numpy()

    # ignore low score
    inds = np.where(scores > 0.6)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = np.argsort(-scores)[:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do nms
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(float, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K fater NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)

    result_data = dets[:, :5].tolist()

    return result_data


def image_process(im, device):
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    # covert BGR to RGB
    im = im[:, :, ::-1]
    im = np.array(im).astype(int)
    im_width, im_height = im.shape[1], im.shape[0]
    scale = [im_width, im_height, im_width, im_height]
    scale = torch.from_numpy(np.array(scale))
    scale = scale.float()
    scale = scale.to(device)
    im -= (104, 117, 123)
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im).unsqueeze(0)
    im = im.float()
    im = im.to(device)
    return im, im_width, im_height, scale


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos
