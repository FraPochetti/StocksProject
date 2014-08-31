# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:10:24 2014

@author: francesco
"""
import numpy as np
import matplotlib.pyplot as plt

txt = open("ModelSelection.txt", "r")

lines = txt.readlines()

accuracies = [float(line[15:-1]) for line in lines if line.startswith('Accuracy')]

txt.close()

shape = np.sqrt(len(accuracies)/3)
## matrix: rows -> time lags, columns -> delta return and moving average
rfc = np.asanyarray(accuracies[0::3]).reshape((shape,shape))
print ''
print rfc
par = np.unravel_index(rfc.argmax(), rfc.shape)
print 'Max RF Accuracy: ' + str(rfc.max()) + ' at ', par

knn = np.asanyarray(accuracies[1::3]).reshape((shape,shape))
print ''
print knn
par = np.unravel_index(knn.argmax(), knn.shape)
print 'Max KNN Accuracy: ' + str(knn.max()) + ' at ', par


#svm = np.asanyarray(accuracies[2::3]).reshape((shape,shape))
#print ''
#print svm
#for ac in svm[:,1]:
#    print ac
#print 'Max SVM Accuracy: ' + str(svm.max()) + ' at ', par
#

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
