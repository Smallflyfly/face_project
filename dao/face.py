#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/22
"""
from sqlalchemy import Column, BIGINT, String, Text, PickleType
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Face(Base):
    __tablename__ = 'base_face'
    id = Column(BIGINT, primary_key=True, comment='id')
    user_id = Column(BIGINT)
    name = Column(String(50))
    feature1 = Column(PickleType)
    feature2 = Column(PickleType)
    image_url = Column(String(255))

