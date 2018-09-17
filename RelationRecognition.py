# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     RelationRecognition
   Description :
   Author :       nicole
   date：          2018/9/12
-------------------------------------------------
   Change Activity:
                   2018/9/12:
-------------------------------------------------
"""
__author__ = 'nicole'

import pymysql
import re

# 打开数据库连接（ip/数据库用户名/登录密码/数据库名）
conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='123456',db="stackexchange")
# 使用cursor()方法创建一个游标对象cursor
cursor=conn.cursor()
# sql语句
sql="select body from posts "

try:
    # 执行sql语句
    body=cursor.execute(sql)
    # data=cursor.fetchone() 获取一条数据
    results=cursor.fetchall()
    for i in results:
        print(re.sub('[^a-zA-Z0-9]', "", "".join(list(i))))
    # 提交到数据库执行
    conn.commit()

    print(body)
except:
    # 如果发生错误则回滚
    conn.rollback()
finally:
    cursor.close()
    conn.close()
    print("数据库断开连接！")


