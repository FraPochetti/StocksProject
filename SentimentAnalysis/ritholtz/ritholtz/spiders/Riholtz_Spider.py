# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 16:07:30 2014

@author: francesco
"""

import scrapy

from ritholtz.items import RitholtzItem

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    #print 'ok1'
    first = [url.strip() for url in urls]
    #print len(first)    
    return first
    
    

class RitholtzSpider(scrapy.Spider):
    name = "ritholtz" ### name of the spider

    allowed_domains = ["alephblog.com"]
    url_list = getCleanStartUrlList('/home/francesco/Dropbox/DSR/StocksProject/financeCrawler/financeCrawler/spiders/ritholtz.txt')
    start_urls = url_list

    def parse(self, response):
        item = RitholtzItem()
 
        item['date'] = response.xpath('//p[@class = "byline"]').re('\w+ \d+\w+, \d+')
        item['keywords'] = response.xpath('//meta[@name = "keywords"]/@content').extract()
        item['body'] = response.xpath('//div[@class = "post-content"]/p | //div[@class = "post-content"]/blockquote').extract()
        yield item
