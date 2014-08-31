# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:08:41 2014

@author: francesco
"""

from scrapy.selector import HtmlXPathSelector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.http import Request
from myspider.items import MachineItem
import urlparse


class MachineSpider(CrawlSpider):
    name = 'myspider'
    allowed_domains = ['example.com']
    start_urls = ['http://www.example.com/index.php']

    rules = (
        Rule(SgmlLinkExtractor(allow_domains=('example.com'),allow=('12\.html'),unique=True),callback='parsecatpage'),
        )

    def parsecatpage(self, response):
        hxs = HtmlXPathSelector(response)
#this works, next line doesn't   categories = hxs.select('//a[contains(@href, "filter=c:Grinders")]')  
        categories = hxs.select('//a[contains(@href, "filter=c:Grinders") or contains(@href, "filter=c:Lathes")]')
        for cat in categories:
            item = MachineItem()
            req = Request(urlparse.urljoin(response.url,''.join(cat.select("@href").extract()).strip()),callback=self.parsetypepage)
            req.meta['item'] = item
            req.meta['machinecategory'] = ''.join(cat.select("./text()").extract())
            yield req

    def parsetypepage(self, response):
        hxs = HtmlXPathSelector(response)
#this works, next line doesn't   types = hxs.select('//a[contains(@href, "filter=t:Disc+-+Horizontal%2C+Single+End")]')
        types = hxs.select('//a[contains(@href, "filter=t:Disc+-+Horizontal%2C+Single+End") or contains(@href, "filter=t:Lathe%2C+Production")]')
        for typ in types:
            item = response.meta['item']
            req = Request(urlparse.urljoin(response.url,''.join(typ.select("@href").extract()).strip()),callback=self.parsemachinelist)
            req.meta['item'] = item
            req.meta['machinecategory'] = ': '.join([response.meta['machinecategory'],''.join(typ.select("./text()").extract())])
            yield req

    def parsemachinelist(self, response):
        hxs = HtmlXPathSelector(response)
        for row in hxs.select('//tr[contains(td/a/@href, "action=searchdet")]'):
            item = response.meta['item']
            req = Request(urlparse.urljoin(response.url,''.join(row.select('./td/a[contains(@href,"action=searchdet")]/@href').extract()).strip()),callback=self.parsemachine)
            print urlparse.urljoin(response.url,''.join(row.select('./td/a[contains(@href,"action=searchdet")]/@href').extract()).strip())
            req.meta['item'] = item
            req.meta['descr'] = row.select('./td/div/text()').extract()
            req.meta['machinecategory'] = response.meta['machinecategory']
            yield req

    def parsemachine(self, response):
        hxs = HtmlXPathSelector(response)
        item = response.meta['item']
        item['machinecategory'] = response.meta['machinecategory']
        item['comp_name'] = 'Name'
        item['description'] = response.meta['descr']
        item['makemodel'] = ' '.join([''.join(hxs.select('//table/tr[contains(td/strong/text(), "Make")]/td/text()').extract()),''.join(hxs.select('//table/tr[contains(td/strong/text(), "Model")]/td/text()').extract())])
        item['capacity'] = hxs.select('//tr[contains(td/strong/text(), "Capacity")]/td/text()').extract()
        relative_image_url = hxs.select('//img[contains(@src, "custom/modules/images")]/@src')[0].extract()
        abs_image_url = urlparse.urljoin(response.url, relative_image_url.strip())
        item['image_urls'] = [abs_image_url]
        yield item

SPIDER = MachineSpider()    