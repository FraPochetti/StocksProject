# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 23:14:41 2014

@author: francesco
"""
import scrapy
import urllib
#from scrapy.selector import HtmlXPathSelector
# start = 'http://www.businessweek.com/archive/news.html#r=404'

def businessWeekUrl():
    totalWeeks = []
    totalPosts = []
    url = 'http://www.businessweek.com/archive/news.html#r=404'
    data = urllib.urlopen(url).read()
    hxs = scrapy.Selector(text=data)

    months = hxs.xpath('//ul/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news.html')
    #months = hxs.xpath('//ul/li/a').re('http://www.businessweek.com/archive/2014-08/news.html')    
    admittMonths = 12*(2013-2007) + 8
    months = months[:admittMonths]
 #   response = scrapy.http.Response(start)
 #   head = scrapy.Selector(response)
 #   months = head.xpath('//ul/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news.html')
    for month in months:
        #print month
        data = urllib.urlopen(month).read()
        hxs = scrapy.Selector(text=data)
        weeks = hxs.xpath('//ul[@class="weeks"]/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news/day\\d+\.html')
        totalWeeks += weeks
    
    for week in totalWeeks:
        #print week
        data = urllib.urlopen(week).read()
        hxs = scrapy.Selector(text=data)
        posts = hxs.xpath('//ul[@class="archive"]/li/h1/a/@href').extract()
        totalPosts += posts
    #print len(totalPosts)
    #print totalPosts[0]
    #print totalPosts[-1]
    
    with open("businessweek.txt", "a") as myfile:
        for post in totalPosts:
            post = post + '\n'
            myfile.write(post)

businessWeekUrl()