#encoding:utf-8
import re
import pymysql

#databases connection
db = pymysql.connect("localhost", "root", "1234", "stackoverflow")
#db = pymysql.connect("localhost", "root", "123456", "text1")
cursor = db.cursor()
cursor.execute("select Tags from Posts limit 50000")
rows = [item[0] for item in cursor.fetchall()]
db.close()

#obtain  two tags write tag-sentence.txt
def get_two_tags():
    tag_list = []
    tag_list_two = []
    tag_two_count = 0
    #print(tag_list)
    with open("tag_sentences.txt", "w") as tags_file:
        for row in rows:
            tag_sentence = str(row)
            tag_sentence = re.sub("<|>", " ", tag_sentence)
            #print(tag_sentence)
            if tag_sentence != "None":
                tag_list = tag_sentence.split()
                new_str = ' '
                #print(tag_list1)
                #print(temp_tag_list)
                if row.count('>')==2:
                    tags_file.write("%s\n" % new_str.join(tag_list))
                    tag_list_two.append(tag_list)
                    tag_two_count +=1
    return tag_two_count,tag_list_two
            #print(tag_list1)
        #         for tag in temp_tag_list:
        #             tag = re.sub("-\d.*$", "", tag)
        #             if not (tag in tag_list):
        #                 tag_list.append(tag)
        # for tag in tag_list:
        #     tags_file.write("%s\n" % tag)

#obtain total databses write tag.txt
def get_total_tags():
    with open("tag.txt", "w") as tags_file:
        for row in rows:
            tag_sentence = str(row)
            tag_sentence = re.sub("<|>", " ", tag_sentence)

            #print(tag_sentence)
            if tag_sentence != "None":
                temp_tag_list = tag_sentence.split()
                new_str = ' '
                tags_file.write("%s\n" % new_str.join(temp_tag_list))

#read total tag databases
def read_tags():
    tag_list_total = []
    with open("tag.txt", "r") as tag_syn:
        for line in tag_syn:
            line = re.sub(r"\t|\n", " ", line)
            line = line.strip(' ')
            tag_list_total += re.split(r" |,", line)
    return tag_list_total
    #print (tag_list_syn)
#print(tag_list_total)
#print(tag_list1)

#obtain tag's type
def get_tag_kind():
    tag_kind = []
    tag_list_total = read_tags()
    for tag in tag_list_total:
        if not (tag in tag_kind):
            tag_kind.append(tag)
    return tag_kind
#print(tag_kind)

#obtain one tag count
def get_one_tag_count():
    tag_dick = {}
    tag_kind = get_tag_kind()
    tag_list_total = get_total_tags()
    for tag in tag_kind:
        count = 0
        tag_dick.fromkeys(tag)
        for tags in tag_list_total:
            if tag == tags:
                count +=1
        tag_dick[tag] = count
#print(tag_dick)

def get_two_tags_count():

    #obtain two tag count
    with open("tag_double_count.txt", "w") as tags_file:
        tag_double_list = []
        temp_double = []
        tag_two_count, tag_list_two = get_two_tags()
        for tag in tag_list_two:
            count = 0
            for tags in tag_list_two:
                if tag[0] == tags[0]:
                    if tag[1] == tags[1]:
                        count +=1
            tag.append("".join('%s')%count)
            tag_double_list.append(tag)
        for tag in tag_double_list:
            if not tag in temp_double:
                temp_double.append(tag)
                new_str = ' '
                tags_file.write("%s\n" % new_str.join(tag))
    return temp_double

def get_tag_frequent():

    with open("tag_frequent.txt", "w") as tags_file:
        tag_two_count, tag_list_two = get_two_tags()
        temp_double = get_two_tags_count()
        for tag in temp_double:
            two_tag_odd = 0.0
            if int(tag[2]) > 4:
                two_tag_odd = int(tag[2]) / tag_two_count
                if two_tag_odd > 0.1:
                    #print(tag)
                    #tags_file.write("%s\n" % new_str.join(tag))
                    new_str = ' '
                    tags_file.write("%s\n" % new_str.join(tag))

get_tag_frequent()