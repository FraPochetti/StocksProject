#python -m pdb project.py
# coding: utf-8

#import Quandl
#import datetime
import pandas as pd
import pandas.io.data
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import glob


#/home/francesco/Dropbox/DSR/Project/datasets
def loadDatasets(path_directory): 
    """
    import into data frame all datasets saved in path_directory
    """
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
#    import glob
#    import re   
#    path = path_directory + "/*.csv"    
#    for files in glob.glob(path):
#        print files
#        dataset_name = re.findall("/(\w+)\.csv", files)[0]
#        dataset_name     = pd.read_csv('sp.csv', index_col=0, parse_dates=True)

#loadDatasets('/home/francesco/Dropbox/DSR/Project/datasets')


######### FEATURE ENGINEERING


### example on what happens for the first dataset sp --> first is an index in range 3:10
### meaning that when first = 3 then delay_return will range from 2:3 which is a list containing only 2.
### This means that the program will add a feature to the sp data frame computing the return of the stock
### respect to 2 days before. Only one feature will be added to the already generated ones to each dataset.
### In the second go of the loop first = 4 meaning that delay_return will range from 2:4 which is a list containing only 2,3.
### The programm will add 2 features to each dataset computing the return of 2 and 3 days before respect to today.
delays = range(2,10)
for first in range(3,7):

### datasets to be loaded

    sp = pd.read_csv('sp.csv', index_col=0, parse_dates=True)
    nasdaq = pd.read_csv('nasdaq.csv', index_col=0, parse_dates=True)
    djia = pd.read_csv('djia.csv', index_col=0, parse_dates=True)
    treasury = pd.read_csv('treasury.csv', index_col=0, parse_dates=True)
    hkong = pd.read_csv('hkong.csv', index_col=0, parse_dates=True)
    frankfurt = pd.read_csv('frankfurt.csv', index_col=0, parse_dates=True)
    paris = pd.read_csv('paris.csv', index_col=0, parse_dates=True)
    nikkei = pd.read_csv('nikkei.csv', index_col=0, parse_dates=True)
    london = pd.read_csv('london.csv', index_col=0, parse_dates=True)
    australia = pd.read_csv('australia.csv', index_col=0, parse_dates=True)

    datasets = [sp, nasdaq, djia, treasury, hkong, frankfurt, paris, nikkei, london, australia]

### function to count NaN values
    def count_missing(frame):
        return (frame.shape[0] * frame.shape[1]) - frame.count().sum()
    
### adding relevant features to the single stock dataset before merging
### delay_return is an numeric array with the number of days I have to go
### bacj in time in order to compute the result. For example a delay = 3 means
### that I will add to the S&P dataset (for example) a feature (Ri - Ri-3)/Ri-3 
#datasets = [sp, nasdaq, treasury, hkong, frankfurt, paris, nikkei, london, australia]

    delay_return = range(2,first)
    print 'Previous days of return accounted: ', delay_return
    

    for dataset in datasets:        
        columns = dataset.columns
        colname = columns[-2] 
        for i in delay_return:
            name = columns[-2][9:] + "Time" + str(i)
            dataset[name] = dataset[colname].pct_change(i)
            
            nameRol = columns[-1][7:] + "RolMean" + str(i)
            dataset[nameRol] = pd.rolling_mean(dataset[columns[-1]], i)
        dataset = dataset.iloc[max(delay_return):,:]

## Merging --> markets dataset from 6th column on because the 6th is the return
## column and the following ones are the delayed returns meaning ther returns
## of i day respect to the i-delta day

    to_be_merged = [nasdaq.iloc[:,6:],
                    djia.iloc[:,6:],
                    treasury.iloc[:,6:],
                    hkong.iloc[:,6:],
                    frankfurt.iloc[:,6:],
                    paris.iloc[:,6:],
                    nikkei.iloc[:,6:],
                    london.iloc[:,6:],
                    australia.iloc[:,6:]]

#to_be_merged = [nasdaq[['Return_Nasdaq']],
#                treasury[['Return_Treasury']],
#                hkong[['Return_HKong']],
#                frankfurt[['Return_Frankfurt']],
#                paris[['Return_Paris']],
#                nikkei[['Return_Nikkei']],
#                london[['Return_London']],
#                australia[['Return_Australia']],
#                oil[['Delta_Oil']],
#                gold[['Delta_Gold']],
#                euro[['Delta_Euro']],
#                yen[['Delta_Yen']],
#                aud[['Delta_Aud']]]
                 
    finance = sp.iloc[:,6:].join(to_be_merged, how = 'outer')

### Cleaning and NaN Imputing

    print 'Size of data frame: ' + str(finance.shape)

    print 'Number of NaN after merging: ' + str(count_missing(finance))
    print 'Percentage of NaN after merging: ' + str(float(count_missing(finance))/(finance.shape[0]*finance.shape[1]))
#######
    finance = finance.interpolate(method='time')
    print 'Number of NaN after time interpolation: ' + str(finance.shape[0]*finance.shape[1] - finance.count().sum())

    finance = finance.fillna(finance.mean())
    print 'Number of NaN after mean interpolation: ' + str(finance.shape[0]*finance.shape[1] - finance.count().sum())

### Temporally Shifting

######### shifting S&P backwards one day in order to have the return 
######### of today matched with the return of yesterday of the other predictors

    shiftBack = -1
    finance.Return_SP500 = finance.Return_SP500.shift(shiftBack)

#print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'

### shifting all the one day Return variables

#### shifting temporally all return features in the range second


    #delays = range(2,6)
   
    maxDelay = max(delays)

    columns = finance.columns[::(2*max(delay_return)-1)]##maxDelay
    for column in columns:
        for delay in delays:
            newcolumn = column + str(delay)
            finance[newcolumn] = finance[column].shift(delay)

    finance = finance.iloc[maxDelay:-1,:]
    #print finance.columns[:7]
    print 'Number of NaN after temporal shifting: ' + str(finance.shape[0]*finance.shape[1] - finance.count().sum())
    #print finance.head()
    print 'Size of data frame after variable creation: ' + str(finance.shape)
##############################################################################
########### PERFORM REGRESSION OR CLASSIFICATION?
    #target = 'REGRESSION'
    target = 'CLASSIFICATION'
#############################################################################

    if target == 'CLASSIFICATION':
        print 'Performing Classification'
    
    #### generating categorical feature to predict
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        
        finance['UpDown'] = finance['Return_SP500']
        finance.UpDown[finance.UpDown >= 0] = 'Up'
        finance.UpDown[finance.UpDown < 0] = 'Down'
        finance.UpDown = le.fit(finance.UpDown).transform(finance.UpDown)
        features = finance.columns[1:-1]
        #print features
    ### splitting in train and test set
        index = int(np.floor(finance.shape[0]*0.8))
        train, test = finance[:index], finance[index:]
        print 'Size of train set: ', train.shape
        print 'Size of test set: ', test.shape    
    
    ##### RANDOM FOREST
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        forest = forest.fit(train[features], train['UpDown'])
        #forestPredicted = forest.predict(test[features])
        #accuracyForest = forest.score(test[features], test['UpDown'].values)
        #forTable = pd.crosstab(test['UpDown'], forestPredicted, rownames=['actual'], colnames=['preds'])
        #print forTable
        #accuracyForest = float(forTable['Up']['Up'] + forTable['Down']['Down'])/forTable.sum().sum()
        print forest.score(test[features],test['UpDown'])
        #print 'Accuracy for Random Forest Classifier: ', accuracyForest
    
    ##### KNN
        from sklearn import neighbors
        model = neighbors.KNeighborsClassifier()
        model.fit(train[features], train['UpDown']) 
        print 'Accuracy for KNN: ', model.score(test[features],test['UpDown'])
    
    ##### SVM
        from sklearn import svm
        mod = svm.SVC()
        mod.fit(train[features], train['UpDown'])  
        #print mod.fit
        print 'Accuracy for SVM: ', mod.score(test[features], test['UpDown'])
        
        print ''
        print ''    
    
    elif target == 'REGRESSION':
        print 'Performing Regression'
        
        features = finance.columns[2:]
        index = int(np.floor(finance.shape[0]*0.8))
        train, test = finance[:index], finance[index:]    
        #print features
        #print test.iloc[:,1]
        #print train.shape, test.shape, test.iloc[:,1].shape      
        
        from sklearn.metrics import mean_squared_error, r2_score
    
    ##### Random Forest
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(n_estimators=100)
        forest = forest.fit(train[features], train.iloc[:,1])
        forestPredicted = forest.predict(test[features])
        print mean_squared_error(test.iloc[:,1],forestPredicted)
        print r2_score(test.iloc[:,1], forestPredicted)
        print ''
        print ''  
#
##rep = 10
##performance = np.zeros(shape=(rep,finance.shape[1]-2))
##                       
##accuracy = np.zeros(rep)
####print features
##for i in range(rep): 
##    forest = RandomForestClassifier(n_estimators=100)
##    ##finance.UpDown
##    ##y, _ = pd.factorize(train['UpDown'])
##    forest = forest.fit(train[features], train['UpDown'])
##    forestPredicted = forest.predict(test[features])
##    forTable = pd.crosstab(test['UpDown'], forestPredicted, rownames=['actual'], colnames=['preds'])
##    ##finance['UpDown'] = pd.Categorical(fina
##    ##cnce.target, finance.target_names)
##    ##print tb
##    #print forestPredicted
##    accuracy[i] = float(forTable['Up']['Up'] + forTable['Down']['Down'])/forTable.sum().sum()
##    #print 'Accuracy: ' +  str(float(forTable['Up']['Up'] + forTable['Down']['Down'])/forTable.sum().sum())
##    #print forTable#
##
##    importances = forest.feature_importances_
##    #indices = np.argsort(importances)[::-1]
##    performance[i,:] = forest.feature_importances_
##
##print 'Accuracy: ' + str(accuracy.mean()) + '\n'
###print(performance)
##importances = performance.mean(axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(finance.shape[1]-2):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

##from sklearn import svm
###from sklearn import linear_model
####model = linear_model.LogisticRegression(C=1e6)
##model = svm.NuSVC()
##model.fit(train[features], train['UpDown'])  
##modelPred = model.predict(test[features])
##modelTable = pd.crosstab(test['UpDown'], modelPred, rownames=['actual'], colnames=['preds'])
##print float(modelTable['Up']['Up'] + modelTable['Down']['Down'])/modelTable.sum().sum()
###modelTable

#from sklearn import neighbors
#
#k = 15
#RMSE = np.zeros(k) 
#R2 = np.zeros(k)
#for p in range(1,k):
#    ### classification
#    model = neighbors.KNeighborsClassifier(p)
#    model.fit(train[features], train['UpDown'])    
#    
#    ### regression    
#    #model = neighbors.KNeighborsRegressor(p)
#    #model.fit(train[features], train.iloc[:,1])  
#    modelPred = model.predict(test[features])
#    #RMSE[p-1] = mean_squared_error(test.iloc[:,1],modelPred)
#    #R2[p-1] = r2_score(test.iloc[:,1],modelPred)
#    modelTable = pd.crosstab(test['UpDown'], modelPred, rownames=['actual'], colnames=['preds'])
#    print float(modelTable['Up']['Up'] + modelTable['Down']['Down'])/modelTable.sum().sum()
#    ###modelTable

#fig, ax = plt.subplots(1,2)
#ax[0].plot(RMSE)
#ax1.plot(RMSE)
#ax[0].legend(('RMSE',))
#ax[1].plot(R2)
#ax[1].legend(('R2',))
#plt.show()


############### CORRELATIONS 
###axs = pd.tools.plotting.scatter_matrix(finance, diagonal='kde')
##
###def wrap(txt, width=8):
###    '''helper function to wrap text for long labels'''
###    import textwrap
###    return '\n'.join(textwrap.wrap(txt, width))
##
###for ax in axs[:,0]: # the left boundary
###    ax.grid('off', axis='both')
###    ax.set_ylabel(wrap(ax.get_ylabel()), rotation=0, va='center', labelpad=20)
###    ax.set_yticks([])
##
###for ax in axs[-1,:]: # the lower boundary
###    ax.grid('off', axis='both')
###    ax.set_xlabel(wrap(ax.get_xlabel()), rotation=70)
###    ax.set_xticks([])
##
###pd.scatter_matrix(finance, diagonal='kde', figsize=(10, 10));
#
###corr = finance.corr()
####from matplotlib.artist import setp
####setp(x.get_xticklabels(), rotation=90)
###plt.xticks(rotation=70)
###plt.imshow(corr, cmap='hot', interpolation='none')
###plt.colorbar()
###plt.xticks(range(len(corr)), corr.columns)
###plt.yticks(range(len(corr)), corr.columns);
