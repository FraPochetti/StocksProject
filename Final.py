# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:22:45 2014

@author: francesco
"""

import functions
import datetime

def final():
    target = 'CLASSIFICATION'
    lags = range(2, 3)
    print 'Maximum time lag applied', max(lags)
    
    start = datetime.datetime(1990, 1, 1)
    end = datetime.datetime(2014, 8, 31)
    out = functions.getStock('GE', start, end)
    datasets = functions.loadDatasets('/home/francesco/Dropbox/DSR/StocksProject/longdatasets')
    datasets.insert(0, out)    
    
    delta = range(2, 5)
    print 'Max Delta days accounted: ', max(delta)
    
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            functions.addFeatures(dataset, adjclose, returns, n)
        #dataset = dataset.iloc[max(delta):,:]
    finance = functions.mergeDataframes(datasets, 6, target)
    #finance = finance.ix[max(delta):]
    print 'Size of data frame: ', finance.shape
    print 'Number of NaN after merging: ', functions.count_missing(finance)
    print '% of NaN after merging: ', (functions.count_missing(finance)/float(finance.shape[0]*finance.shape[1]))*100, '%'
    
    finance = finance.interpolate(method = 'time')
    print 'Number of NaN after time interpolation: ', functions.count_missing(finance)

    finance = finance.fillna(finance.mean())
    print 'Number of NaN after mean interpolation: ', functions.count_missing(finance)    

    back = -1
    #finance.Return_SP500 = finance.Return_SP500.shift(back)
    finance.Return_Out = finance.Return_Out.shift(back)
    
    finance = functions.applyTimeLag(finance, lags, delta, back, target)
    #finance = functions.mergeSentimenToStocks(finance)
    #print finance.columns
    print 'Number of NaN after temporal shifting: ', functions.count_missing(finance)
    print 'Size of data frame after feature creation: ', finance.shape
    if target == 'CLASSIFICATION':
        start_test = datetime.datetime(2014,4,1)
        X_train, y_train, X_test, y_test  = functions.prepareDataForClassification(finance, start_test)
        
        print ''
        #print 'Performing CV...'
        #grid = {'n_estimators': [80, 100, 150], 'learning_rate': [0.01, 0.1, 1, 10]}
        #grid = {'n_estimators': [50, 80, 100, 1000]}
        #functions.performTimeSeriesSearchGrid(finance, 4, 0.8, features, 'ADA', grid)        
        
        print functions.performClassification(X_train, y_train, X_test, y_test, 'RF', [])

if __name__ == "__main__":
    final()     