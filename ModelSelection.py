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
    import datetime

    target = 'CLASSIFICATION'
    lags = range(2, maxlag) 
    print 'Maximum time lag applied', max(lags)
    print ''

    for maxdelta in range(3,12):
        #datasets = functions.loadDatasets('/home/francesco/Dropbox/DSR/StocksProject/longdatasets')
        #start = datetime.datetime(1990, 1, 1)
        #end = datetime.datetime(2014, 8, 31)
        #out = functions.getStock('AAPL', start, end)
        datasets = functions.loadDatasets('/home/francesco/Dropbox/DSR/StocksProject/longdatasets')
        #datasets.insert(0, out)


        delta = range(2,maxdelta) 
        print 'Delta days accounted: ', max(delta)
    
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
    
        finance = finance.interpolate(method='time')
        print 'Number of NaN after time interpolation: ', functions.count_missing(finance)

        finance = finance.fillna(finance.mean())
        print 'Number of NaN after mean interpolation: ', functions.count_missing(finance)    
        
        back = -1
        finance.Return_Out = finance.Return_Out.shift(back)

        finance = functions.applyTimeLag(finance, lags, delta, back, target)
    
        print 'Number of NaN after temporal shifting: ', functions.count_missing(finance)
    
        print 'Size of data frame after feature creation: ', finance.shape   
    
        if target == 'CLASSIFICATION':
            start_test = datetime.datetime(2014,4,1)
            X_train, y_train, X_test, y_test  = functions.prepareDataForClassification(finance, start_test)
         
            acc = functions.performCV(X_train, y_train, 10, 'GTB', [])           
            print ''            
            print 'Mean Accuracy for (%d, %d): %f' % (max(lags), max(delta), acc)             
            #print functions.performClassification(X, y, X_val, y_val, 'ADA', [100, 1])
            print '============================================================================'

if __name__ == '__main__':
    for i in range(1,2):
        import sys
        sys.stdout = open('./ClassificRes/Procter/GTB%s.txt' %str(i), 'w')
        for maxlag in range(3,12):
            performFeatureSelection(maxlag)