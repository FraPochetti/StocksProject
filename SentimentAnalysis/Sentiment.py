# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:45:19 2014

@author: francesco
"""
import re
import pandas as pd
from sklearn import preprocessing

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
    
    if filename == '/home/francesco/BigData/Project/ritho.json':
        def getCorrectDate(wrongdate):
            mon_day_year = re.search( r'(\w+) (\d+)\w+, (\d+)', wrongdate)
            month, day, year = mon_day_year.group(1), mon_day_year.group(2), mon_day_year.group(3)
            return month + ' ' + day + ' ' + year
            
        df['date'] = df['date'].apply(getCorrectDate)
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = df['date'].apply(lambda x: x[:10])
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.drop_duplicates(subset = ['keywords'])
    df = df.sort(columns='date')
    #df = df.set_index('date')
    df['text'] = df['keywords'] + df['body'] 

    df = df.drop('body', 1)
    df = df.drop('keywords', 1)
    
    return df
    

def cleanText(text):
    """
    removes punctuation, stopwords and returns lowercase text in a list of single words
    """
    text = text.lower()    
    
    from bs4 import BeautifulSoup
    text = BeautifulSoup(text).get_text()
    
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    
    from nltk.corpus import stopwords
    clean = [word for word in text if word not in stopwords.words('english')]
    
    return clean

def loadPositive():
    """
    loading positive dictionary
    """
    #C:\Users\Fra\Dropbox\DSR\StocksProject\LoughranMcDonald_Positive.csv
    #/home/francesco/Dropbox/DSR/StocksProject/LoughranMcDonald_Positive.csv
    myfile = open('/home/francesco/Dropbox/DSR/StocksProject/LoughranMcDonald_Positive.csv', "r")
    positives = myfile.readlines()
    positive = [pos.strip().lower() for pos in positives]
    return positive

def loadNegative():
    """
    loading positive dictionary
    """
    #C:\Users\Fra\Dropbox\DSR\StocksProject\LoughranMcDonald_Negative.csv
    #/home/francesco/Dropbox/DSR/StocksProject/LoughranMcDonald_Negative.csv
    myfile = open('/home/francesco/Dropbox/DSR/StocksProject/LoughranMcDonald_Negative.csv', "r")
    negatives = myfile.readlines()
    negative = [neg.strip().lower() for neg in negatives]
    return negative
    
def countNeg(cleantext, negative):
    """
    counts negative words in cleantext
    """
    negs = [word for word in cleantext if word in negative]
    return len(negs)

def countPos(cleantext, positive):
    """
    counts negative words in cleantext
    """
    pos = [word for word in cleantext if word in positive]
    return len(pos)   
       
    
def getSentiment(cleantext, negative, positive):
    """
    counts negative and positive words in cleantext and returns a score accordingly
    """
    positive = loadPositive()
    negative = loadNegative()
    return (countPos(cleantext, positive) - countNeg(cleantext, negative))
    
def updateSentimentDataFrame(df):
    """
    performs sentiment analysis on single text entry of dataframe and returns dataframe with scores
    """
    positive = loadPositive()
    negative = loadNegative()   
    
    df['text'] = df['text'].apply(cleanText)
    df['score'] = df['text'].apply(lambda x: getSentiment(x,negative, positive))
    #clean = pd.Series([cleanText(text) for text in list(df['text'])])    
    #df['text'] = clean
    return df

def prepareToConcat(filename):
    """
    load a csv file and gets a score for the day
    """
    df = pd.read_csv(filename, parse_dates=['date'])
    df = df.drop('text', 1)
    df = df.dropna()
    df = df.groupby(['date']).mean()
    name = re.search( r'/(\w+).csv', filename)
    df.columns.values[0] = name.group(1)
    return df

def createSentimentDataset(sentimentdata):
    """
    merges the available datasets passed as a list into one single sentiment 
    """
    df = sentimentdata[0].join(sentimentdata[1:], how = 'outer')
         
    df = df.fillna(method='bfill')
    df = df.fillna(method='ffill')
    
    i = df.index
    c = df.columns
    
    print df.shape[0]*df.shape[1] - df.count().sum()
    return pd.DataFrame(preprocessing.scale(df), index = i, columns=c)

def recreateUniqueBlogDataset(listOfFiles):
    """
    takes as argument a list of datasets (cleaned using the function prepareToConcat
    to be merged row wise in order to recreate one single data frame for a financial blog
    in case more csv files were produced for single blog
    """
    df = pd.read_csv(listOfFiles[0], parse_dates=['date'])    
    for data in listOfFiles[1:]:
        second = pd.read_csv(data, parse_dates=['date'])        
        df = df.append(second)
    df = df.drop_duplicates(subset = ['text'])
    df = df.drop('text', 1)
    df = df.dropna()
    df = df.groupby(['date']).mean()
    name = re.search( r'/(\w+).csv', listOfFiles[0])
    df.columns.values[0] = name.group(1)
    return df
    
    

print 'Reading json'    
df = readJson('/home/francesco/BigData/Project/businsider3.json')
print 'Performing Sentiment...'
updateSentimentDataFrame(df).to_csv('/home/francesco/BigData/Project/CSV/businsider3.csv', index = False)

#aleph = prepareToConcat('/home/francesco/BigData/Project/CSV/aleph.csv')
#busweek = prepareToConcat('/home/francesco/BigData/Project/CSV/busweek.csv')
#ritho = prepareToConcat('/home/francesco/BigData/Project/CSV/ritho.csv')
#tech = recreateUniqueBlogDataset(['/home/francesco/BigData/Project/CSV/tech2.csv', '/home/francesco/BigData/Project/CSV/tech.csv'])
#businside = recreateUniqueBlogDataset(['/home/francesco/BigData/Project/CSV/businsider1.csv', '/home/francesco/BigData/Project/CSV/businsider2.csv', '/home/francesco/BigData/Project/CSV/businsider3.csv'])
#sentiment = createSentimentDataset([aleph, busweek, ritho, tech, businside])
#sentiment.to_csv('/home/francesco/BigData/Project/CSV/sentiment.csv', index = True)




