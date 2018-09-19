# -*-coding:UTF-8-*-
import nltk
import pymysql

try:
    #获取数据库连接
    conn = pymysql.connect(host='localhost',user='root',passwd='123456',db='test',charset='utf8')
    #打印数据库连接对象
    print('数据库连接对象为：{}'.format(conn))
    # 获取游标
    cur = conn.cursor()

   
    #   ----------进行查询操作-----------------------
    #  #打印游标
    # #print("游标为：{}".format(cur))
    # #print("游标是："+cur)
    # print("游标是：" + str(cur))
    # #查询sql语句
    # sql='select * from project_jizhan limit 0,1'
    # cur.execute(sql)
    # conn.commit()
    # #使用 fetchall() 方法获取数据对象
    # #data = cur.fetchall()
    # #使用 fetchone() 方法获取一条数据
    # data = cur.fetchone()
    # for item in data:
    #     print(item)
    with conn:
        # 仍然是，第一步要获取连接的cursor对象，用于执行查询
        cur = conn.cursor()
        # 类似于其他语言的query函数，execute是python中的执行查询函数
        cur.execute("select Body from posts")
        # 使用fetchall函数，将结果集（多维元组）存入rows里面
        rows = cur.fetchall()
        # 依次遍历结果集，发现每个元素，就是表中的一条记录，用一个元组来显示
        from nltk.tokenize import sent_tokenize
        i=0
        for row in rows:
            #print(row)
            #sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            "".join(list(row))
            rr = str(tuple(row))
            all_sent = sent_tokenize(rr)
            #print(all_sent)
            "".join(list(all_sent))
            tt=str(tuple(all_sent))
            str1=tt.replace(', \'', '\n')
            print(str1)
            #print(tt)
            #print(sentences)
    cur.close()
    conn.close()
except Exception as e:
            print(e)
