# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:31:07 2014

@author: francesco
"""
def loadDatasets(path_directory): 
    """
    import into dataframe all datasets saved in path_directory
    """
    import pandas as pd
    
    name = path_directory + '/sp.csv'
    sp = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/nasdaq.csv'
    nasdaq = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/djia.csv'
    djia = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/treasury.csv'
    treasury = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/hkong.csv'
    hkong = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/frankfurt.csv'
    frankfurt = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/paris.csv'
    paris = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/nikkei.csv'
    nikkei = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/london.csv'
    london = pd.read_csv(name, index_col=0, parse_dates=True)
    
    name = path_directory + '/australia.csv'
    australia = pd.read_csv(name, index_col=0, parse_dates=True)
    
    return [sp, nasdaq, djia, treasury, hkong, frankfurt, paris, nikkei, london, australia]


def count_missing(dataframe):
    """
    count number of NaN in dataframe
    """
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

    
def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days

    """
    import pandas as pd
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)
    
def mergeDataframes(datasets, index):
    """
    merges datasets in the list 
    """
    subset = []
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    
    #to_be_merged = [nasdaq.iloc[:, index:], djia.iloc[:, index:], treasury.iloc[:, index:],
    #                hkong.iloc[:, index:], frankfurt.iloc[:,index:], paris.iloc[:, index:],
    #                nikkei.iloc[:,index:], london.iloc[:,index:], australia.iloc[:, index:]]
    return datasets[0].iloc[:, index:].join(subset, how = 'outer')
    
def applyTimeLag(dataset, lags, delta, back):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lads list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:]
    
def performClassification(dataset, split):
   """
   performs classification on returns using serveral algorithms
   """
   from sklearn import preprocessing
   import numpy as np
   
   le = preprocessing.LabelEncoder()
        
   dataset['UpDown'] = dataset['Return_SP500']
   dataset.UpDown[dataset.UpDown >= 0] = 'Up'
   dataset.UpDown[dataset.UpDown < 0] = 'Down'
   dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
   features = dataset.columns[1:-1]
   
   index = int(np.floor(dataset.shape[0]*split))
   train, test = dataset[:index], dataset[index:]
   print 'Size of train set: ', train.shape
   print 'Size of test set: ', test.shape
   
   performRFClass(train, test, features)
   
   performKNNClass(train, test, features)
   
   performSVMClass(train, test, features)
    
    
def performRegression():
    pass

    
def performRFClass(train, test, features):
    """
    Random Forest Binary Classification
    """
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    forest = forest.fit(train[features], train['UpDown'])
    print 'Accuracy RFC: ', forest.score(test[features],test['UpDown'])
        
def performKNNClass(train, test, features):
    """
    KNN binary Classification
    """
    from sklearn import neighbors
    model = neighbors.KNeighborsClassifier()
    model.fit(train[features], train['UpDown']) 
    print 'Accuracy KNN: ', model.score(test[features],test['UpDown'])

def performSVMClass(train, test, features):
    """
    SVM binary Classification
    """
    from sklearn import svm
    mod = svm.SVC()
    mod.fit(train[features], train['UpDown'])  
    print 'Accuracy SVM: ', mod.score(test[features], test['UpDown'])    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    