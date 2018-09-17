# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ExtractTags
   Description :
   Author :       nicole
   date：          2018/9/13
-------------------------------------------------
   Change Activity:
                   2018/9/13:
-------------------------------------------------
"""
__author__ = 'nicole'

import pymysql
import nltk
import re

# 信息提取模型分5步：分句，用nltk.sent_tokenize(text)实现,得到一个list of strings
# 分词，[nltk.word_tokenize(sent) for sent in sentences]实现，得到list of lists of strings
# 标记词性，[nltk.pos_tag(sent) for sent in sentences]实现得到一个list of lists of tuples
# 前三步可以定义在一个函数中
def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences
# 实体识别，关系识别

# 打开数据库连接（ip/数据库用户名/登录密码/数据库名）
conn=pymysql.connect(host='localhost',port=3306,user='root',passwd='123456',db="stackexchange")
# 使用cursor()方法创建一个游标对象cursor
cursor=conn.cursor()
# sql语句
sql="select distinct tags from posts where tags is not null"

try:
    # 执行sql语句
    body=cursor.execute(sql)
    # data=cursor.fetchone() 获取一条数据
    results=cursor.fetchall()
    for i in results:
        print(re.sub('[^a-zA-Z0-9]', "", "".join(list(i))))


    # 提交到数据库执行
    conn.commit()
    # 打印查询个数
    print(body)
except:
    # 如果发生错误则回滚
    conn.rollback()
finally:
    cursor.close()
    conn.close()
    print("数据库断开连接！")



