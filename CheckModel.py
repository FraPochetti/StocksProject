# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:10:24 2014

@author: francesco
"""
#import numpy as np
#import matplotlib.pyplot as plt
#import operator

import re

txt = open("./ClassificRes/GeneralElectric/RF1.txt", "r")
lines = txt.readlines()
accuracies = [float(re.search( r': (0.\d+)', line).group(1)) for line in lines if line.startswith('Mean')]
txt.close()
print 'Max RF: ' + str(max(accuracies))

txt = open("./ClassificRes/GeneralElectric/QDA1.txt", "r")
lines = txt.readlines()
accuracies = [float(re.search( r': (0.\d+)', line).group(1)) for line in lines if line.startswith('Mean')]
txt.close()
print 'Max QDA: ' + str(max(accuracies))

txt = open("./ClassificRes/GeneralElectric/KNN1.txt", "r")
lines = txt.readlines()
accuracies = [float(re.search( r': (0.\d+)', line).group(1)) for line in lines if line.startswith('Mean')]
txt.close()
print 'Max KNN: ' + str(max(accuracies))

txt = open("./ClassificRes/GeneralElectric/SVM1.txt", "r")
lines = txt.readlines()
accuracies = [float(re.search( r': (0.\d+)', line).group(1)) for line in lines if line.startswith('Mean')]
txt.close()
print 'Max SVM: ' + str(max(accuracies))

txt = open("./ClassificRes/GeneralElectric/ADA1.txt", "r")
lines = txt.readlines()
accuracies = [float(re.search( r': (0.\d+)', line).group(1)) for line in lines if line.startswith('Mean')]
txt.close()
print 'Max ADA: ' + str(max(accuracies))

txt = open("./ClassificRes/GeneralElectric/GTB1.txt", "r")
lines = txt.readlines()
accuracies = [float(re.search( r': (0.\d+)', line).group(1)) for line in lines if line.startswith('Mean')]
txt.close()
print 'Max GTB: ' + str(max(accuracies))

# 
#knn = np.asanyarray(accuracies[1::nm]).reshape((shape,shape))
#print ''
#print ''
#print 'KNN'
#print knn
#par = np.unravel_index(knn.argmax(), knn.shape)
#print 'Max KNN Accuracy: ' + str(knn.max()) + ' at ', par
#
#
#svm = np.asanyarray(accuracies[2::nm]).reshape((shape,shape))
#print ''
#print ''
#print 'SVM'
#print svm
#for ac in svm[:,1]:
#    print ac
#print 'Max SVM Accuracy: ' + str(svm.max()) + ' at ', par
#
#ada = np.asanyarray(accuracies[1::nm]).reshape((shape,shape))
#print ''
#print ''
#print 'ADA BOOSTING'
#print ada
#par = np.unravel_index(ada.argmax(), ada.shape)
#print 'Max Ada Boosting Accuracy: ' + str(ada.max()) + ' at ', par
#
#gtb = np.asanyarray(accuracies[1::nm]).reshape((shape,shape))
#print ''
#print ''
#print 'GRADIENT TREE BOOSTING'
#print gtb
#par = np.unravel_index(gtb.argmax(), gtb.shape)
#print 'Max Gradient Tree Boosting Accuracy: ' + str(gtb.max()) + ' at ', par

#f, axarr = plt.subplots(int(shape), 1, sharex=True, sharey=True)
#for j in range(9):
#    axarr[j].plot(range(2,11), rfc[j,:], color='b', linewidth=2.5)
#    axarr[j].plot(range(2,11), knn[j,:], color='r', linewidth=2.5)
#    axarr[j].plot(range(2,11), svm[j,:], color='g', linewidth=2.5)
#    axarr[j].legend(('RF', 'KNN', 'SVM'),loc='center left', bbox_to_anchor = (1.0, 0.5))
#
#f.subplots_adjust(hspace=0)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.show()
