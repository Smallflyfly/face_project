#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/11/10
"""
import collections
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
from models.retinaface import RetinaFace
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
device = torch.device("cuda")

input_frame = collections.deque()
output_frame = collections.deque()

resnet = InceptionResnetV1(pretrained='vggface2', device=None).eval()
transform = transforms.Compose(
    [transforms.ToTensor()]
)


def detection(im):
    resize = 1
    im, im_width, im_height, scale = image_process(im, device)
    loc, conf, landms = retina_net(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data


def capture():
    capture = cv2.VideoCapture(1)
    while True:
        _, frame = capture.read()
        # input_frame.append(frame)
        face_result = detection(frame)
        for det in face_result:
            xmin, ymin, xmax, ymax, conf = det
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
            im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im_crop = im_pil.crop(box=(xmin, ymin, xmax, ymax))
            tic = time.time()
            feature_in = generate_feature(im_pil)
            print('feature generate: ', time.time() - tic)
            array_in = string2array(feature_in)
            torch_in_feature = torch.from_numpy(array_in).cuda().unsqueeze(0)
            name = match(torch_in_feature)
            if not name:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            fontStyle = ImageFont.truetype(
                "font/FZY1JW.TTF", 20, encoding="utf-8"
            )
            draw.text((xmin, ymin - 20), name, font=fontStyle)
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
            output_frame.append(frame)

        # cv2.imshow('im', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # capture.release()
    # cv2.destroyAllWindows()


def generate_feature(im):
    im = im.resize((128, 128))
    im_tensor = transform(im)
    img_embedding = resnet(im_tensor.unsqueeze(0))[0]
    features = img_embedding.cpu().detach().numpy().tostring()
    return features


def string2array(feature):
    array = np.frombuffer(feature, dtype=np.float32)
    return array


def match(torch_in_feature):
    mysqldb = MySQLDB()
    session = mysqldb.session()
    faces = session.query(Face).all()
    max_similarity = 999999.0
    name = None
    tic = time.time()
    for face in faces:
        feature_db = face.feature1
        array_db = string2array(feature_db)
        torch_db_feature = torch.from_numpy(array_db).cuda().unsqueeze(0)
        cos_similarity = torch.pairwise_distance(torch_in_feature, torch_db_feature)
        if cos_similarity.cpu().detach().numpy()[0] < max_similarity:
            name = face.name
            max_similarity = cos_similarity.cpu().detach().numpy()[0]
            # print(max_similarity)
    print('match time: ', time.time() - tic)
    if max_similarity > 1.1:
        return None
    else:
        return name


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
    print(len(output_frame))
    while True:
        if output_frame:
            frame = output_frame.popleft()
            cv2.imshow('im', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # url = 'rtsp://admin:fang2831016@172.27.12.188:554/stream1'
    # capture()
    capture_process = multiprocessing.Process(target=capture)
    capture_process.start()
    # sleep(1)
    show()
    # show_process = multiprocessing.Process(target=show)
    # show_process.start()
    # capture = cv2.VideoCapture(1)
    # while True:
    #     _, frame = capture.read()
    #     # input_frame.append(frame)
    #     face_result = detection(frame)
    #     for det in face_result:
    #         xmin, ymin, xmax, ymax, conf = det
    #         xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    #         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
    #         im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #         im_crop = im_pil.crop(box=(xmin, ymin, xmax, ymax))
    #         tic = time.time()
    #         feature_in = generate_feature(im_pil)
    #         print('feature generate: ', time.time() - tic)
    #         array_in = string2array(feature_in)
    #         torch_in_feature = torch.from_numpy(array_in).cuda().unsqueeze(0)
    #         name = match(torch_in_feature)
    #         if not name:
    #             continue
    #         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #         draw = ImageDraw.Draw(image)
    #         fontStyle = ImageFont.truetype(
    #             "font/FZY1JW.TTF", 20, encoding="utf-8"
    #         )
    #         draw.text((xmin, ymin - 20), name, font=fontStyle)
    #         frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    #         output_frame.append(frame)

        # cv2.imshow('im', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cv2.destroyAllWindows()
