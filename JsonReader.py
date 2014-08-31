# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:45:19 2014

@author: francesco
"""

def readJson(filename):
    """
    reads a json file and returns a clean pandas data frame
    """
    import pandas as pd
    df = pd.read_json(filename)
    
    def unlist(element):
        return ''.join(element)
    
    for column in df.columns:
        df[column] = df[column].apply(unlist)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(cols='keywords')
    df = df.sort(columns='date')
    df = df.set_index('date')
    
    return df.head(3)
    
    
print readJson('/home/francesco/BigData/Project/businessweek.json')