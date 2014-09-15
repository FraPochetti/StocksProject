# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 23:14:41 2014

@author: francesco
"""
import scrapy
import urllib

#### list of blogs:
# businessweek.com
# http://www.businessinsider.com/archives?date=2007-07-09&vertical=
# http://alephblog.com/
# http://pragcap.com/
# http://www.ritholtz.com/blog/
# http://www.dailyworth.com/
# http://www.wisebread.com/
# http://www.thereformedbroker.com/
# http://wealthpilgrim.com/


#from scrapy.selector import HtmlXPathSelector
# start = 'http://www.businessweek.com/archive/news.html#r=404'

def businessWeekUrl():
    totalWeeks = []
    totalPosts = []
    url = 'http://www.businessweek.com/archive/news.html#r=404'
    data = urllib.urlopen(url).read()
    hxs = scrapy.Selector(text=data)

    months = hxs.xpath('//ul/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news.html')
    admittMonths = 12*(2013-2007) + 8
    months = months[:admittMonths]
 
    for month in months:
        
        data = urllib.urlopen(month).read()
        hxs = scrapy.Selector(text=data)
        weeks = hxs.xpath('//ul[@class="weeks"]/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news/day\\d+\.html')
        totalWeeks += weeks
    
    for week in totalWeeks:
        data = urllib.urlopen(week).read()
        hxs = scrapy.Selector(text=data)
        posts = hxs.xpath('//ul[@class="archive"]/li/h1/a/@href').extract()
        totalPosts += posts
 
    with open("businessweek.txt", "a") as myfile:
        for post in totalPosts:
            post = post + '\n'
            myfile.write(post)



def businessInsiderUrl():
    
    from datetime import date, datetime, timedelta
    totalPosts = []
    
    def perdelta(start, end, delta):
        curr = start
        while curr < end:
            yield curr
            curr += delta
            
    totalDays = ['http://www.businessinsider.com/archives?vertical=businessinsider&date='+str(day) for day in perdelta(date(2008, 1, 2), date(2014, 8, 16), timedelta(days=1))]
    
    for day in totalDays:
        data = urllib.urlopen(day).read()
        hxs = scrapy.Selector(text=data)
        totalPosts += ['http://www.businessinsider.com/'+url for url in hxs.xpath('//td/a/@href').extract()]

    with open("businessinsider.txt", "a") as myfile:
        for post in totalPosts[:30]:
            print post
            post = post + '\n'
            myfile.write(post)        
    
    
def alephBlogUrl():
    
    totalPosts = []
    
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']    
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014']
    totalMonths = [ 'http://alephblog.com/' + year + '/' + month + '/page/' for year in years for month in months]
    totalMonths = totalMonths[:-4]
    
    for month in totalMonths:
        page = 1
        new = ['noError']
        while len(new) > 0:                   
            temp = month + str(page)
            #print temp
            data = urllib.urlopen(temp).read()
            hxs = scrapy.Selector(text=data)
            new = hxs.xpath('//div[@class = "post"]/h3/a/@href').extract()
            totalPosts += new
            page += 1
            
    with open("aleph.txt", "a") as myfile:
        for post in totalPosts:
            post = post + '\n'
            myfile.write(post)        
    

def ritholtzBlogUrl():
    
    totalPosts = []
    
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']    
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014']
    totalMonths = [ 'http://ritholtz.com/blog/' + year + '/' + month + '/page/' for year in years for month in months]
    totalMonths = totalMonths[:-4]
    
    for month in totalMonths:
        page = 1
        new = ['noError']
        while len(new) > 0:                   
            temp = month + str(page)
            print temp
            data = urllib.urlopen(temp).read()
            hxs = scrapy.Selector(text=data)
            new = hxs.response.xpath('//div[@class = "headline"]/h2/a/@href').extract()
            totalPosts += new
            page += 1
            
    with open("ritholtz.txt", "a") as myfile:
        for post in totalPosts:
            post = post + '\n'
            myfile.write(post)        
    
    print len(totalPosts)
    print(totalPosts[0])
    print(totalPosts[len(totalPosts)-1])


def TechCrunchBlogUrl():
    
    totalPosts = []
 
    totalPages = [ 'http://techcrunch.com/page/' + str(page) + '/' for page in range(2415,5741)]
    
    for page in totalPages:
        print page
        data = urllib.urlopen(page).read()
        hxs = scrapy.Selector(text=data)
        totalPosts += hxs.xpath('//li[@class = "river-block"]/@data-permalink').extract()
        #print hxs.xpath('//li[@class = "river-block"]/@data-permalink').extract()
            
    with open("techcrunch2.txt", "a") as myfile:
        for post in totalPosts:
            try:
                post = post + '\n'
                myfile.write(post)
            except:
                pass
    
# date response.xpath('//meta[@name="sailthru.date"]/@content').extract()
# keywords response.xpath('//meta[@name="sailthru.title"]/@content').extract()
#body response.xpath('//div[@class = "article-entry text"]/p | //div[@class = "post-content"]/blockquote').extract()


TechCrunchBlogUrl()
    


