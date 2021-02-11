# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:16:59 2019

@author: Yogendra
"""

data = open('test.txt','w')
data.write('Hello world')
print(data)
data.close()