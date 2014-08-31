# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:01:47 2014

@author: francesco
"""

import scrapy

from financeCrawler.items import FinancecrawlerItem

def getCleanStartUrlList(filename):
    myfile = open(filename, "r")
    urls = myfile.readlines()
    return [url.strip() for url in urls]
    
    

class BWSpider(scrapy.Spider):
    name = "busweek"
    allowed_domains = ["businessweek.com"]
    url_list = getCleanStartUrlList('businessweek.txt')
    start_urls = url_list    
    
    def parse(self, response):
        item = FinancecrawlerItem()
        item['date'] = response.xpath('//meta[@content][@name="pub_date"]/@content').extract()
        item['keywords'] = response.xpath('//meta[@content][@name="keywords"]/@content').extract() 
        item['body'] = response.xpath('//div[@id = "article_body"]/p/text()').extract()
        yield item
        
        
