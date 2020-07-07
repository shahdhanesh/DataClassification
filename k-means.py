from pylab import plot,show
from numpy import vstack,array, int32
from numpy.random import rand
# from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import hungarian
import data_handler

filename = "HandWrittenLetters.txt"
number_per_class = 39
test_instances = "1:3"
nu_clusters = 26

trainX, trainY, testX, testY = data_handler.splitData2TestTrain(filename, number_per_class, test_instances)

trainX.extend(testX)

trainX = array(trainX, int32)
testX = array(testX, int32)
# colors = ["g.","r.","c.","y.", "w.", "b.", "a.", "e.", "f.", "h."]
kmeans = KMeans (n_clusters=nu_clusters)
kmeans.fit(trainX)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# for i in range(len(trainX)):
    # print("coordinate:",trainX[i], "label:", labels[i])
    # plt.plot(trainX[i][0], trainX[i][1],"bo", markersize = 10)

# plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
# plt.show()

predictions = kmeans.predict(testX)
predictions = [str(item) for item in predictions]

cm = confusion_matrix(testY, predictions)

acc = hungarian.cluster_acc(testY, predictions)
print(cm)
print(acc)