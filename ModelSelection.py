# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:51:22 2014

@author: francesco
"""

def performModelSelection(maxlag):
    from temp import *


    target = 'CLASSIFICATION'
    #target = 'REGRESSION'

    lags = range(2, maxlag)
    print 'Maximum time lag applied', max(lags)
    print ''

    for maxdelta in range(3,12):
    
        datasets = loadDatasets('/home/francesco/Dropbox/DSR/Project/datasets')

        delta = range(2,maxdelta)
        print 'Delta days accounted: ', max(delta)
    
        for dataset in datasets:
            columns = dataset.columns    
            adjclose = columns[-2]
            returns = columns[-1]
            for n in delta:
                addFeatures(dataset, adjclose, returns, n)
            dataset = dataset.iloc[max(delta):,:] # computation of returns and moving means introduces NaN which are nor removed
    
        finance = mergeDataframes(datasets, 6)
    
        print 'Size of data frame: ', finance.shape
        print 'Number of NaN after merging: ', count_missing(finance)
    
        finance = finance.interpolate(method='time')
        print 'Number of NaN after time interpolation: ', finance.shape[0]*finance.shape[1] - finance.count().sum()

        finance = finance.fillna(finance.mean())
        print 'Number of NaN after mean interpolation: ', (finance.shape[0]*finance.shape[1] - finance.count().sum())    

        back = -1
        finance.Return_SP500 = finance.Return_SP500.shift(back)
    
        finance = applyTimeLag(finance, lags, delta, back)
    
        print 'Number of NaN after temporal shifting: ', count_missing(finance)
    
        print 'Size of data frame after feature creation: ', finance.shape   
    
        if target == 'CLASSIFICATION':
            performClassification(finance, 0.8)
            print ''
    
        elif target == 'REGRESSION':
            performRegression(finance, 0.8)
            print ''

import sys
sys.stdout = open('ModelSelection.txt', 'w')
           
for maxlag in range(3,12):
    performModelSelection(maxlag)