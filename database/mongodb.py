#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/{DAY}
"""
import pymongo


class Mongodb(object):
    def __init__(self):
        self.host = '111.229.203.174'
        self.username = 'fang'
        self.password = '123456'
        self.port = '27017'
        self.database = 'facedb'
        # self.uri = 'mongodb://fang:123456@111.229.203.174:27017/?authSource=facedb&authMechanism=SCRAM-SHA-1'
        self.uri = 'mongodb://{}:{}@{}:{}/?authSource={}&authMechanism=SCRAM-SHA-1' \
            .format(self.username, self.password, self.host, self.port, self.database)
        # print(self.uri)
        self.client = pymongo.MongoClient(self.uri)
