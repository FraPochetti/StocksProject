# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 19:24:44 2014

@author: francesco
"""

import scrapy

from aleph.items import AlephItem

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    #print 'ok1'
    first = [url.strip() for url in urls]
    #print len(first)    
    return first
    
    

class AlephSpider(scrapy.Spider):
    name = "aleph" ### name of the spider

    allowed_domains = ["alephblog.com"]
    url_list = getCleanStartUrlList('/home/francesco/Dropbox/DSR/StocksProject/financeCrawler/financeCrawler/spiders/aleph.txt')
    start_urls = url_list

    def parse(self, response):
        item = AlephItem()
 
        item['date'] = response.xpath('//head/meta[@property = "article:published_time"]/@content').extract()
        item['keywords'] = response.xpath('//head/title/text()').extract()
        item['body'] = response.xpath('//div[@class = "entry"]/p').extract()
        yield item
