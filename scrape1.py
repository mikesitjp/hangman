#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:03:21 2023

@author: masayasusaito
"""

import urllib.request
from bs4 import BeautifulSoup

class Scraper:
    def __init__(self,site):
        self.site = site

    def scrape(self):
        r = urllib.request.urlopen(self.site)        
        soup = BeautifulSoup(r.read(), 'html.parser')      
        with open('output.txt','w') as f:
            for tag in soup.find_all('a'):
                url = tag.get('href')
                if url is None:
                    continue
                print('\n'+url)
                f.write(url+'\n')
                
news = 'https://news.google.com/'
Scraper(news).scrape()


