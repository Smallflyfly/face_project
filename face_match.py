#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/03/23
"""
import time

import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageDraw, ImageFont

from dao.face import Face
from data import cfg_mnet
from database.mysqldb import MySQLDB
from models.resnet import Resnet18Triplet
from models.retinaface import RetinaFace
from utils.net_utils import load_model, image_process, process_face_data
import numpy as np


checkpoint = torch.load('weights/model_resnet18_triplet.pt')
model = Resnet18Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
best_distance_threshold = checkpoint['best_distance_threshold']

flag_gpu_available = torch.cuda.is_available()

if flag_gpu_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

cfg = cfg_mnet
retina_trained_model = './weights/mobilenet0.25_Final.pth'
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net = load_model(retina_net, retina_trained_model, False)
retina_net = retina_net.cuda(0)
retina_net.eval()

preprocess = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(size=224),  # Pre-trained model uses 224x224 input images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6068, 0.4517, 0.3800],  # Normalization settings for the model, the calculated mean and std values
        std=[0.2492, 0.2173, 0.2082]  # for the RGB channels of the tightly-cropped VGGFace2 face dataset
    )
])


# img = cv2.imread('face.jpg')  # Or from a cv2 video capture stream

# Note that you need to use a face detection model here to crop the face from the image and then
#  create a new face image object that will be inputted to the facial recognition model later.

# Convert the image from BGR color (which OpenCV uses) to RGB color
# img = img[:, :, ::-1]
#
# img = preprocess(img)
# img = img.unsqueeze(0)
# img = img.to(device)
#
# embedding = model(img)
#
# # Turn embedding Torch Tensor to Numpy array
# embedding = embedding.cpu().detach().numpy()
# print(embedding)

mysqldb = MySQLDB()
session = mysqldb.session()
faces = session.query(Face).all()


def detection(im):
    resize = 1
    im, im_width, im_height, scale = image_process(im, device)
    loc, conf, landms = retina_net(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data


def capture():
    # url = 'rtsp://admin:fang2831016@172.27.12.188:554/stream1'
    capture = cv2.VideoCapture(0)
    while True:
        _, frame = capture.read()
        face_result = detection(frame)
        for det in face_result:
            xmin, ymin, xmax, ymax, conf = det
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
            im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im_W, im_H = im_pil.size
            xmin_, ymin_, xmax_, ymax_ = max(0, xmin-20), max(0, ymin-20), min(im_W-1, xmax+20), min(im_H-1, ymax+20)
            im_crop = im_pil.crop(box=(xmin_, ymin_, xmax_, ymax_))
            tic = time.time()
            features = generate_feature(im_crop)
            print('feature generate: ', time.time() - tic)
            array_in = string2array(features)
            torch_in_feature = torch.from_numpy(array_in).cuda()
            max_similarity = -999
            name = None
            for face in faces:
                feature_db = face.feature1
                array_db = string2array(feature_db)
                torch_db_feature = torch.from_numpy(array_db).cuda()
                # torch.cosine_similarity [-1, 1]
                cos_similarity = torch.cosine_similarity(torch_in_feature, torch_db_feature, dim=0)
                # cos_similarity = torch.pairwise_distance(torch_in_feature, torch_db_feature)
                if cos_similarity.cpu().detach().numpy() > max_similarity:
                    name = face.name
                    max_similarity = cos_similarity.cpu().detach().numpy()
            # print(name)
            name = name if name is not None else "陌生人"
            if name is not None:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                fontStyle = ImageFont.truetype(
                    "font/FZY1JW.TTF", 20, encoding="utf-8"
                )
                draw.text((xmin, ymin - 20), name, font=fontStyle)
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)

        cv2.imshow('face-detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def generate_feature(im):
    if im.mode != 'RGB':
        im = im.convert('RGB')
    # im_tensor = transform(im)
    # print(im_tensor.shape)
    im = preprocess(im)
    im = im.unsqueeze(0).to(device)
    img_embedding = model(im)[0]
    features = img_embedding.cpu().detach().numpy().tostring()
    return features


def string2array(feature):
    array = np.frombuffer(feature, dtype=np.float32)
    return array


if __name__ == '__main__':
    capture()
