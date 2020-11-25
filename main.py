#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/{DAY}
"""
import io
import time
import uuid

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from fastapi import FastAPI, UploadFile, File
from gridfs import GridFS
from torchvision import transforms

from dao.face import Face
from database.mongodb import Mongodb
from database.mysqldb import MySQLDB
from models.face_detection import face_detection

app = FastAPI(title='人脸管理接口')
mongodb = Mongodb()
db = mongodb.client.facedb

transform = transforms.Compose([
    transforms.Resize(size=(160, 160)),
    transforms.ToTensor()
]
)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


@app.post("/face/upload", description='人脸库上传 只支持单张单人图片')
async def uploadFile(file: UploadFile = File(...)):
    contents = await file.read()
    gfs = GridFS(db, collection='face')
    file_id = gfs.put(contents, content_type='image/jpeg', filename=file.filename)
    im_pil, cv_im = init_image(contents)
    tic = time.time()
    dets = face_detection(cv_im)
    print(time.time() - tic)
    if len(dets) == 0:
        return {"code": 400, "success": False, "message": "未检测出人脸，请重新上传"}
    det = dets[0]
    boxes, score = det[:4], det[4]
    im_pil = im_pil.crop([boxes[0], boxes[1], boxes[2], boxes[3]])
    print(im_pil.size)
    features = generate_feature(im_pil)
    mysqldb = MySQLDB()
    session = mysqldb.session()
    name = file.filename[:file.filename.index('.')]
    face = session.query(Face).filter(Face.name == name).scalar()
    if face:
        face.feature1 = features
    else:
        face = Face()
        face.user_id = str(uuid.uuid1())
        face.name = name
        face.feature1 = features
        face.image_url = str(file_id)
        session.add(face)
    session.commit()
    session.close()

    return {"code": 200, "success": True, "file_id": str(file_id)}


@app.get("/getFile")
async def getFile(file_id):
    gfs = GridFS(db, collection='face')
    image_file = gfs.find_one(file_id)
    print(image_file)
    return file_id


@app.post("/face/match", description='上传图片 人脸比对 返回最相似的姓名')
async def faceMatch(file: UploadFile = File(...)):
    contents = await file.read()
    im_pil, cv_im = init_image(contents)
    tic = time.time()
    dets = face_detection(cv_im)
    print('face detecion time: ', time.time() - tic)
    if len(dets) == 0:
        return {"code": 400, "success": False, "message": "未检测出人脸，请重新上传"}
    det = dets[0]
    boxes, score = det[:4], det[4]
    im_pil = im_pil.crop([boxes[0], boxes[1], boxes[2], boxes[3]])
    tic = time.time()
    feature_in = generate_feature(im_pil)
    print('feature generate: ', time.time() - tic)
    array_in = string2array(feature_in)
    torch_in_feature = torch.from_numpy(array_in).cuda().unsqueeze(0)
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
        # cos_similarity = torch.cosine_similarity(torch_in_feature, torch_db_feature, dim=0)
        cos_similarity = torch.pairwise_distance(torch_in_feature, torch_db_feature)
        if cos_similarity.cpu().detach().numpy()[0] < max_similarity:
            name = face.name
            max_similarity = cos_similarity.cpu().detach().numpy()[0]

    # print(max_similarity)
    if max_similarity > 1.1:
        return {"code": 400, "success": False, "message": "未匹配到"}
    print('match time: ', time.time() - tic)
    return {"code": 200, "success": True, "name": name}


def generate_feature(im):
    # im = im.resize((160, 160))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    print(im.mode)
    im_tensor = transform(im)
    print(im_tensor.shape)
    img_embedding = resnet(im_tensor.unsqueeze(0))[0]
    features = img_embedding.cpu().detach().numpy().tostring()
    return features


def init_image(contents):
    content = io.BytesIO(contents)
    im_pil = Image.open(content)
    w, h = im_pil.size
    max_ = max(w, h)
    if max_ > 1280:
        new_w = int(w / max_ * 1280)
        new_h = int(h / max_ * 1280)
        im_pil = im_pil.resize((new_w, new_h))

    cv_im = cv2.cvtColor(np.asarray(im_pil), cv2.COLOR_RGB2BGR)
    print(cv_im.shape)
    return im_pil, cv_im


def string2array(feature):
    array = np.frombuffer(feature, dtype=np.float32)
    return array

