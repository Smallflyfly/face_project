#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/09/22
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class MySQLDB(object):
    def __init__(self):
        self.host = 'cdb-1yzvsfly.bj.tencentcdb.com'
        self.username = 'root'
        self.password = 'fang2831016'
        self.port = '10006'
        self.database = 'company'
        self.url = "mysql+pymysql://{}:{}@{}:{}/{}".format(self.username, self.password,
                                                           self.host, self.port, self.database)
        self.engine = create_engine(self.url, echo=False)
        self.session = sessionmaker(bind=self.engine, expire_on_commit=False)
