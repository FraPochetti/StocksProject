# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 08:00:33 2014

@author: francesco
"""

from lxml import html
import requests
import json

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    first = [url.strip() for url in urls]  
    return first

def alephSpider():
    url_list = getCleanStartUrlList('/home/francesco/Dropbox/DSR/StocksProject/financeCrawler/financeCrawler/spiders/aleph.txt')
    #url_list = url_list[:3]
    scraped = {'date':[], 'keywords':[], 'body':[]}
    
    for url in url_list:
        print url
        try:
            page = requests.get(url)
            tree = html.fromstring(page.text)
            scraped['date'].append(tree.xpath('//head/meta[@property = "article:published_time"]/@content'))
            scraped['keywords'].append(tree.xpath('//head/title/text()'))
            scraped['body'].append(tree.xpath('//div[@class = "entry"]/p/text()'))
        except:
            pass

    with open('/home/francesco/BigData/Project/aleph.json', 'wb') as fp:
        json.dump(scraped, fp)
    #return scraped

alephSpider()