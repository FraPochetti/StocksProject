# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:22:45 2014

@author: francesco
"""

import functions
import numpy as np

def final():
    target = 'CLASSIFICATION'
    lags = range(2, 16)
    print 'Maximum time lag applied', max(lags)
    datasets = functions.loadDatasets('/home/francesco/Dropbox/DSR/StocksProject/datasets')

    delta = range(2, 31) 
    print 'Max Delta days accounted: ', max(delta)
    
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            functions.addFeatures(dataset, adjclose, returns, n)
        dataset = dataset.iloc[max(delta):,:]
    finance = functions.mergeDataframes(datasets, 6, target)
    print 'Size of data frame: ', finance.shape
    print 'Number of NaN after merging: ', functions.count_missing(finance)
    print '% of NaN after merging: ', (functions.count_missing(finance)/float(finance.shape[0]*finance.shape[1]))*100, '%'
    
    finance = finance.interpolate(method='time')
    print 'Number of NaN after time interpolation: ', functions.count_missing(finance)#finance.shape[0]*finance.shape[1] - finance.count().sum()

    finance = finance.fillna(finance.mean())
    print 'Number of NaN after mean interpolation: ', functions.count_missing(finance)#(finance.shape[0]*finance.shape[1] - finance.count().sum())    

    back = -1
    finance.Return_SP500 = finance.Return_SP500.shift(back)
    
    finance = functions.applyTimeLag(finance, lags, delta, back, target)
    
    print 'Number of NaN after temporal shifting: ', functions.count_missing(finance)
    print 'Size of data frame after feature creation: ', finance.shape
    
    if target == 'CLASSIFICATION':
        finance, features = functions.prepareDataForClassification(finance)
        
        print ''
        print 'Performing CV...'
        grid = {'n_estimators': [80, 100, 150], 'learning_rate': [0.01, 0.1, 1]}
        #grid = {'n_estimators': [50, 80, 100, 1000]}
        functions.performTimeSeriesSearchGrid(finance, 4, 0.8, features, 'ADA', grid)        
        
        
        #functions.performCV(finance, 4, 0.8, features, 'ADA')
    
        #print functions.performClassification(finance, features, 0.8, 'ADA', [100, 1])

#acc = np.zeros(1)
#for i in range(1):
#    acc[i] = final()

#print acc.mean()

final()     