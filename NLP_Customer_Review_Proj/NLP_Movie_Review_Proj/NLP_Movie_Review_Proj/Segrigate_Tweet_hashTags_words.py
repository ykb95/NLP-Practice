# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:30:09 2019

@author: Yogendra
"""

import re
file_data = open("tweets.txt", "r", encoding="utf-8")
#final_output = open('output.txt', 'w', encoding="utf-8")
data = file_data.read()
all_clean_words = []
c1=0 
c2=0
for i in data.split():
    word = i
    c1=c1+1
    if(word[0]=='#'):
       word = i.replace('#','')
       word = re.sub('([A-Z][a-z]+)', r' \1' ,word)
       c2=c2+1
       #final_output.write('\n' + word)
       all_clean_words.append(word)

print(all_clean_words)
#print(c1, c2, len(all_clean_words))       
#final_output.close()
file_data.close()