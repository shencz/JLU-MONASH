#encoding:utf-8
import re

def read_tags():
    tag_list = []
    temp_tag = []
    with open("tag_frequent.txt", "r") as tag_syn:
        for line in tag_syn:
            line = re.sub(r"\t|\n", " ", line)
            line = line.strip(' ')
            tag_list.append(line)
    return tag_list

def get_word():
    tag_list = read_tags()
    word_type_list = []
    with open("processed_data1.txt", "r") as tag_syn:
        with open("word_type.txt", "w") as tags_file:
            for line in tag_syn:
                line = re.sub(r"\t|\n", " ", line)
                line = line.strip(' ')
                word_list = re.split(r" |,", line)
                for tag in tag_list:
                    tag = tag.split()
                    #print(tag)
                    #print(word_list)
                    if (tag[0] in word_list) and (tag[1] in word_list):
                        print(word_list)
                        #word_type_list.append(word_list)
                        if word_list != "None":
                            new_str = ' '
                            tags_file.write("%s\n" % new_str.join(word_list))
    #print(word_type_list)

get_word()