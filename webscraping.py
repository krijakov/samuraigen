#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Description: Assembles a dataset of samurai names from the website : https://samurai-archives.com/wiki/Main_Page

'''

from bs4 import BeautifulSoup
import requests
import re

def scrape(urls):
    """Scrapes the data from the given list of urls and provides some base pattern matching 
    returning a raw list of names"""
    names = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        splits = str(soup).split('<li>')
        for s in splits[1:-1]:
            match = re.search(r'title="(.*?)"', s)
            if match:
                names.append(match.group(1))
    return names

def filterNames(names):
    """Remove duplicates and extra remarks"""

    allnames = []
    for n in sorted(list(set(names))):

        filt_name = []
        for w in n.split(" "):
            w = w.replace("-", " ")
            if w[:2] == "b." or w[0] == "(":
                break
            else:
                filt_name.append(w)
            
        allnames.append(" ".join(filt_name))
    allnames = sorted(list(set(allnames))) # remove duplicates once more in case some were generated after removing the remarks
    return allnames

if __name__ == '__main__':
    
    # generate urls:
    start_from = "A B C D E F G H I J K M N O P R S T U W Y Z".split(" ")
    urls = ["https://samurai-archives.com/w/index.php?title=Category:Samurai&from=" + s for s in start_from]

    names = scrape(urls)
    allnames = filterNames(names)

    # export names:
    with open('Data/samurai_names.txt', 'w') as f:
        f.write('\n'.join(allnames))

    

