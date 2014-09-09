# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:51:22 2014

@author: francesco
performs bidimensional feature selection. The degrees of freedom are :
- lags --> lag days applied i.e. to predict what will happen tomorrow I shift predictor returns of 2,3..max(lags) days.
           meaning that to predict tomorrow I'm using not only yesterday (default) but also what happened 2,3,4-days ago

- delta --> period of time over which to compute return. Default is daily (delta = 1-day). The function creates features 
            computing tomorrow's return VS (not only yesterday) 2,3,4-days ago. Delta are also the days over which to compute 
            the moving average of the prevous returns.
"""

def performFeatureSelection(maxlag):
    import functions   

    target = 'CLASSIFICATION'
    #target = 'REGRESSION'

    lags = range(2, maxlag) 
    print 'Maximum time lag applied', max(lags)
    print ''

    for maxdelta in range(3,8):
        datasets = functions.loadDatasets('/home/francesco/Dropbox/DSR/StocksProject/datasets')

        delta = range(2,maxdelta) 
        print 'Delta days accounted: ', max(delta)
    
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
    
        finance = finance.interpolate(method='time')
        print 'Number of NaN after time interpolation: ', functions.count_missing(finance)

        finance = finance.fillna(finance.mean())
        print 'Number of NaN after mean interpolation: ', functions.count_missing(finance)    

        back = -1
        finance.Return_SP500 = finance.Return_SP500.shift(back)
    
        finance = functions.applyTimeLag(finance, lags, delta, back, target)
    
        print 'Number of NaN after temporal shifting: ', functions.count_missing(finance)
    
        print 'Size of data frame after feature creation: ', finance.shape   
    
        if target == 'CLASSIFICATION':
            functions.performClassification(finance, 0.8)
            print ''
    
        elif target == 'REGRESSION':
            functions.performRegression(finance, 0.8)
            print ''

#import sys
#sys.stdout = open('./RegresRes/RFreg50_50.txt', 'w')

for maxlag in range(3,8):
    performFeatureSelection(maxlag)