
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:01:47 2014

@author: francesco
"""

import scrapy

from businside.items import BusinsideItem

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    #print 'ok1'
    first = [url.strip() for url in urls]
    #print len(first)    
    return first
    
    

class BISpider(scrapy.Spider):
    name = "biweek" ### name of the spider

    allowed_domains = ["businessinsider.com"]
    url_list = getCleanStartUrlList('businessinsider.txt')
    start_urls = url_list[200001:]

    def parse(self, response):
        item = BusinsideItem()
 
        item['date'] = response.xpath('//meta[@content][@name="date"]/@content').extract()
        item['keywords'] = response.xpath('//meta[@content][@name="news_keywords"]/@content').extract()
        item['body'] = response.xpath('//div[@class = "KonaBody post-content"]').extract()
        yield item
        



    
