
import numpy as np
import math
import operator
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import data_handler
import random

data1 = pd.read_csv('ATNTFaceImages400.txt', header = -1).as_matrix()
ATX = np.transpose(data1[1:,:])
ATY = np.transpose(data1[0,:])
new_data = np.transpose(data1)

def get_data():
    return pre_processor(file=open('ATNT50/trainDataXY.txt'))


def pre_processor(file):
    indexes = []
    data_raw = []
    data = []
    for index, line in enumerate(file):
        if index != 0:
            data_raw.append(line.rstrip().rsplit(','))
        else:
            indexes = line.rstrip().rsplit(',')
    for x in range(0, len(indexes)):
        index_list = [sample[x] for sample in data_raw]
        data.append(index_list + [indexes[x]])
    return data


def getCentroid(labelVectors):
    curr_centroid = []
    for j in range(len(labelVectors[0]) - 1):
        curr_sum = 0
        for i in range(len(labelVectors)):
            curr_sum += int(labelVectors[i][j])
        curr_sum = float(curr_sum) / len(labelVectors)
        curr_centroid.append(curr_sum)
    curr_centroid.append(labelVectors[-1][-1])
    return curr_centroid


# return [ float(sum([item[j] for item in labelVectors]))/len(labelVectors) for j in range(len(labelVectors[0]))]


def predict(trainX, trainY, testX, testY, k):
    unique_trainY = list(set(trainY))
    clustersDict = {}
    total_post_count = 0
    predicteds = []

    for x in range(len(trainY)):
        clustersDict.setdefault(trainY[x], []).append(trainX[x])

    for testInstance in testX:
        train_data = [getCentroid(clustersDict[key]) for key in clustersDict.keys()]
        instanceResults = getNeighbours(train_data, testInstance, k)
        predictedClass = mode([item[-1] for item in instanceResults])

        predicteds.append(predictedClass)

        if predictedClass == testInstance[-1]:
            total_post_count += 1
            clustersDict.setdefault(predictedClass).append(testInstance)

    return (float(total_post_count) / len(testX)) * 100.00, predicteds


def mode(numbers):
    largestCount = 0
    modes = []
    for x in numbers:
        if x in modes:
            continue
        count = numbers.count(x)
        if count > largestCount:
            del modes[:]
            modes.append(x)
            largestCount = count
        elif count == largestCount:
            modes.append(x)
    return modes[0]


def train_test_split(data):
    split_randInt = random.randint(int(len(data) / 2), (len(data) - 5))
    random.shuffle(data)
    train_data = data[0:split_randInt]
    test_data = data[split_randInt:-1]
    return train_data, test_data

def getEuclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)


def getNeighbours(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neightbours = []
    for x in range(k):
        neightbours.append(distances[x][0])
    return neightbours


def getResponse(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def classification(classifier):
    nfold_average, average_accuracy = [], []

    kf = KFold(len(ATY), n_folds=5, shuffle=True)
    sumAvgs = 0
    sumAvgs = 0
    if (classifier == "Centroid"):

        data, indexes = data_handler.get_data("ATNTFaceImages400.txt")
        accuracy=cross_validator(5, data, indexes, classifier)



    for train_index, test_index in kf:
        Xtrain, Xtest = ATX[train_index], ATX[test_index]

        ytrain, ytest = ATY[train_index], ATY[test_index]
        trainingSet = []
        testSet = []
        for x in range(len(Xtrain)):
            trainingSet.append([])
            for y in range(len(Xtrain)):
                trainingSet[x].append(Xtrain[x][y])
            trainingSet[x].append(ytrain[x])

        for x in range(len(Xtest)):
            testSet.append([])
            for y in range(len(Xtest)):
                testSet[x].append(Xtest[x][y])
            testSet[x].append(ytest[x])

        """trainingSet = np.concatenate((Xtrain,ytrain), axis = 1)
        testSet = np.concatenate((Xtest,ytest), axis = 1)"""
        if (classifier == "KNN"):
            predictions = []
            k = 5
            for x in range(len(testSet)):
                neighbours = getNeighbours(trainingSet, testSet[x], k)
                result = getResponse(neighbours)
                predictions.append(result)
            accuracy = getAccuracy(testSet, predictions)
            print('Accuracy: ' + repr(accuracy) + '%')

            """knneighbors = KNeighborsClassifier(n_neighbors=5)
            knneighbors.fit(Xtrain, ytrain)
            predictions = knneighbors.predict(Xtest)"""

        elif (classifier == "SVM"):
            svmclassifier = svm.LinearSVC()
            svmclassifier.fit(Xtrain, ytrain)
            predictions = svmclassifier.predict(Xtest)
            actual = ytest
            accuracy = metrics.accuracy_score(actual, predictions) * 100
            print("accuracy with ", classifier, " and 5 folds:", accuracy)

        elif (classifier == "LogisticRegression"):
            regr = linear_model.LogisticRegression()
            regr.fit(Xtrain, ytrain)
            predictions = regr.predict(Xtest)
            actual = ytest
            accuracy = metrics.accuracy_score(actual, predictions) * 100
            print("accuracy with ", classifier, " and 5 folds:", accuracy)

        sumAvgs += accuracy
    avg_acc = sumAvgs/5
    print("Average Accuracy:", avg_acc);
    print("----------------------------------------------")
    """nfold_average.append(avg_acc)
    average_accuracy = average_accuracy + [j for j in nfold_average]"""
    return avg_acc

def cross_validator(k, train_data, feature_names, classifier):
    for index, item in enumerate(train_data):
        item.append(feature_names[index])
    random.shuffle(train_data)
    k_splits = np.array_split(train_data, k)
    feature_splits = [[in_item[-1] for in_item in item]for item in k_splits]
    all_accuracy =  0
    for k in range(0,k):
        print ("For %s fold" %(int(k)+1))
        trainX = []
        trainY = []
        testX = k_splits[k]
        testY = feature_splits[k]
        trainX_temp = k_splits[:k] + k_splits[(k + 1):]
        trainY_temp = feature_splits[:k] + feature_splits[(k + 1):]
        for x in range(len(trainX_temp)):
            trainX.extend(trainX_temp[x])
            trainY.extend(trainY_temp[x])
        accuracy1 = predict(trainX, trainY, testX, testY, 4)
        all_accuracy += accuracy1[0]
        print(accuracy1[0])
    k_accuracy =(all_accuracy)/(int(k)+1)
    print(k_accuracy)
    return all_accuracy


Y_knn = classification("KNN")
Y_svm = classification("SVM")
Y_regress = classification("LogisticRegression")
Y_cent = classification("Centroid")
print()
print("Average of KNN : ", round(Y_knn, 2))
print("Average of Centroid : ", round(Y_cent,2))
print("Average of SVM : ", round(Y_svm,2))
print("Average of linear_model :", round(Y_regress,2))
# ----------------------Plotting Graph-----------------------
"""x = [2,3,5,10]
# Plot the data
plt.scatter(x, Y_knn,color='black',marker='^')
plt.plot(x, Y_knn, label='KNN')
plt.scatter(x, Y_svm,color='black',marker='^')
plt.plot(x, Y_svm, label='SVM')
plt.scatter(x, Y_cent,color='black',marker='^')
plt.plot(x, Y_cent, label='Centroid')
# Add a legend
plt.legend(loc='lower right', frameon=False)
# Show the plot
plt.show()"""