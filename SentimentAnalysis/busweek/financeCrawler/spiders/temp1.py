# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 19:58:18 2014

@author: francesco
"""

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    #print 'ok1'
    first = [url.strip() for url in urls]
    print len(first)
    print first[0]
    print first[len(first)-1]    
    #return first

def myfunction(text):    
    try:
        text = unicode(text, 'utf-8')
    except TypeError:
        return text

print(myfunction(u'\u2019'))       
#getCleanStartUrlList('techcrunch.txt')