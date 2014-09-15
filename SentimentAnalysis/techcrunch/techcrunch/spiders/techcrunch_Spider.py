# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 23:05:10 2014

@author: francesco
"""

import scrapy

from techcrunch.items import TechcrunchItem

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    #print 'ok1'
    first = [url.strip() for url in urls]
    #print len(first)    
    return first
    
    

class techcrunchItemSpider(scrapy.Spider):
    name = "techcrunch" ### name of the spider

    allowed_domains = ["techcrunch.com"]
    url_list = getCleanStartUrlList('/home/francesco/Dropbox/DSR/StocksProject/financeCrawler/financeCrawler/spiders/techcrunch2.txt')
    start_urls = url_list

    def parse(self, response):
        item = TechcrunchItem()
 
        item['date'] = response.xpath('//meta[@name="sailthru.date"]/@content').extract()
        item['keywords'] = response.xpath('//meta[@name="sailthru.title"]/@content').extract()
        item['body'] = response.xpath('//div[@class = "article-entry text"]/p | //div[@class = "post-content"]/blockquote').extract()
        yield item