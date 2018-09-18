# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     017
   Description :
   Author :       nicole
   date：          2018/9/5
-------------------------------------------------
   Change Activity:
                   2018/9/5:
-------------------------------------------------
"""
__author__ = 'nicole'

# import nltk
# from nltk.corpus import udhr
# languages=['Chickasaw', 'English', 'German_Deutsch','Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
# cfd=nltk.ConditionalFreqDist(
#     (lang,len(word))
#     for lang in languages
#     for word in udhr.words(lang+'-Latin1')
# )
# cfd.tabulate(conditions=['English','German_Deutsch'],samples=range(10),cumulative=True)

import nltk
from nltk.corpus import brown
days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)
cfd.tabulate(conditions=days,samples=days,cumulative=True)