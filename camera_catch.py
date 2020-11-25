#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/11/10
"""
import collections
import pickle
import time
from time import sleep

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.backends import cudnn
import multiprocessing

from torchvision import transforms

from dao.face import Face
from data import cfg_re50, cfg_mnet
from database.mysqldb import MySQLDB
from models.resnet import resnet50
from models.retinaface import RetinaFace
from utils import net_utils
from utils.net_utils import load_model, image_process, process_face_data
import numpy as np


# cfg = cfg_re50
cfg = cfg_mnet
retina_trained_model = './weights/mobilenet0.25_Final.pth'
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net = load_model(retina_net, retina_trained_model, False)
retina_net = retina_net.cuda(0)
retina_net.eval()
cudnn.benchmark = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


input_frame = collections.deque()
output_frame = collections.deque()

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

transform = transforms.Compose(
    [transforms.ToTensor()]
)

N_IDENTITY = 8631
feature_model = resnet50(num_classes=N_IDENTITY, include_top=False)
feature_model_weight = './weights/resnet50_scratch_weight.pkl'
net_utils.load_state_dict(feature_model, feature_model_weight)
feature_model.to(device)
feature_model.eval()
print('feature model load successfully!')

mysqldb = MySQLDB()
session = mysqldb.session()
faces = session.query(Face).all()

colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]

mean_bgr = np.array([91.4953, 103.8827, 131.0912])
mean_rgb = np.array([131.0912, 91.4953, 103.8827]).astype(float)


def detection(im):
    resize = 1
    im, im_width, im_height, scale = image_process(im, device)
    loc, conf, landms = retina_net(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data


def generate_feature_resnet(im):
    im = im.resize((224, 224))
    print(im)
    im -= mean_rgb
    print(im)
    im = transform(im).unsqueeze(0).to(device)
    img_embedding = feature_model(im)
    print(img_embedding)
    return img_embedding


def capture():
    url = 'rtsp://admin:fang2831016@172.27.12.188:554/stream1'
    capture = cv2.VideoCapture(url)
    count = 1
    while True:
        _, frame = capture.read()
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        if count % 10 == 0:
            count = 0
            face_result = detection(frame)
            # print(face_result)
            for det in face_result:
                xmin, ymin, xmax, ymax, conf = det
                # xmin, ymin, xmax, ymax = int(xmin) * 4, int(ymin) * 4, int(xmax) * 4, int(ymax) * 4
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
                # 人识别
                im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im_crop = im_pil.crop(box=(xmin, ymin, xmax, ymax))
                # feature_in = generate_feature(im_crop)
                feature_in = generate_feature_resnet(im_crop)
                array_in = string2array(feature_in)
                torch_in_feature = torch.from_numpy(array_in).cuda().unsqueeze(0)
                tic = time.time()
                # name, similarity = match(faces, torch_in_feature)
                # print('match time: {}, name:{}, similarity:{}'.format(time.time() - tic, name, similarity))
                # if not name:
                #     continue
                # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # draw = ImageDraw.Draw(image)
                # fontStyle = ImageFont.truetype(
                #     "font/FZY1JW.TTF", 20, encoding="utf-8"
                # )
                # draw.text((xmin, ymin - 20), name, font=fontStyle)
                # frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        count += 1
        cv2.imshow('im', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def generate_feature(im):
    im = im.resize((160, 160))
    im_tensor = transform(im)
    img_embedding = resnet(im_tensor.unsqueeze(0).cuda())[0]
    features = img_embedding.cpu().detach().numpy().tostring()
    return features


def string2array(feature):
    array = np.frombuffer(feature, dtype=np.float32)
    return array


def match(faces, torch_in_feature):
    tic = time.time()
    min_similarity = 999999.0
    name = None
    for face in faces:
        feature_db = face.feature1
        array_db = string2array(feature_db)
        torch_db_feature = torch.from_numpy(array_db).cuda().unsqueeze(0)
        dist = (torch_in_feature - torch_db_feature).norm().item()
        print(dist)
        cos_similarity = torch.pairwise_distance(torch_in_feature, torch_db_feature)
        if cos_similarity.cpu().detach().numpy()[0] < min_similarity:
            name = face.name
            min_similarity = cos_similarity.cpu().detach().numpy()[0]
    print('match for 循环时间： ', time.time() - tic)
    if min_similarity > 0.9:
        return None, min_similarity
    else:
        return name, min_similarity


def face_detection():
    while input_frame:
        frame = input_frame.popleft()
        face_result = detection(frame)
        for det in face_result:
            xmin, ymin, xmax, ymax, conf = det
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
        cv2.imshow('im', frame)


def show():
    # print(len(output_frame))
    while True and output_frame:
        if output_frame:
            frame = output_frame.popleft()
            cv2.imshow('im', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    capture()
