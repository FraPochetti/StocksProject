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
    urls = getCleanStartUrlList('businessweek.txt')
    start_urls = urls[:10]

    def parse(self, response):
        item = FinancecrawlerItem()
        item['date'] = response.xpath('//meta[@content][@name="pub_date"]/@content').extract()
        item['keywords'] = response.xpath('//meta[@content][@name="keywords"]/@content').extract() 
        item['body'] = response.xpath('//div[@id = "article_body"]/p/text()').extract()
        yield item
#    

## -*- coding: utf-8 -*-
#"""
#Created on Sun Aug 31 10:01:47 2014
#
#@author: francesco
#"""
#
#import scrapy
#
#from financeCrawler.items import FinancecrawlerItem
#
#def getCleanStartUrlList(filename):
#    myfile = open(filename, "r")
#    urls = myfile.readlines()
#    print 'ok1'
#    first = [url.strip() for url in urls]
#    print len(first)    
#    return first
#    
#    
#
#class BWSpider(scrapy.Spider):
#    name = "finance" ### name of the spider
#    print name
#    domain = 'businside' ### domain to pass to the domains dict to return allowed domains and start_urls   
#    print domain
#    
#    if domain == 'busweek':
#         allowed_domains = ["businessweek.com"]
#         url_list = getCleanStartUrlList('businessweek.txt')
#         start_urls = url_list[:2]
#         
#    elif domain == 'businside':
#         print domain 
#         allowed_domains = ["businessinsider.com"]
#         url_list = getCleanStartUrlList('businessinsider.txt')
#         start_urls = url_list[:2]
#         print start_urls
#    
#    
#    def parse(self, response):
#        print 'hello'        
#        print domain
#        #pass
#        item = FinancecrawlerItem()
#        
#        if domain == 'busweek':
#            item['date'] = response.xpath('//meta[@content][@name="pub_date"]/@content').extract()
#            item['keywords'] = response.xpath('//meta[@content][@name="keywords"]/@content').extract() 
#            item['body'] = response.xpath('//div[@id = "article_body"]/p/text()').extract()
#            yield item
#        
#        elif domain == 'businside':
#            item['date'] = response.xpath('//meta[@content][@name="date"]/@content').extract()
#            item['keywords'] = response.xpath('//meta[@content][@name="news_keywords"]/@content').extract()
#            item['body'] = response.xpath('//div[@class = "KonaBody post-content"]').extract()
#            yield item
#        
#
#
#
#    
