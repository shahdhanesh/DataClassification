import numpy as np
import pandas as pd
import matplotlib as mpl
import random
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import math
import operator
import data_handler

data = pd.read_csv('HandWrittenLetters.txt', header=-1).as_matrix()
trainX, trainY, testX, testY = data_handler.splitData2TestTrain(data_handler.pickDataClass('Handwrittenletters.txt', data_handler.letter_2_digit_convert("abcde")), 39, "1:9")


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

    return (float(total_post_count) / len(testX)) * 100.00


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

    trainingSetA = data[:, 0:30]
    testSetA = data[:, 30:39]
    trainingSetB = data[:, 39:69]
    testSetB = data[:, 69:78]
    trainingSetC = data[:, 78:108]
    testSetC = data[:, 108:117]
    trainingSetD = data[:, 117:147]
    testSetD = data[:, 147:156]
    trainingSetE = data[:, 156:186]
    testSetE = data[:, 186:195]

    A_X_train = data[1:, 0:30]
    A_X_test = data[1:, 30:39]

    A_y_train = data[0:1, 0:30]
    A_y_test = data[0:1, 30:39]

    B_X_train = data[1:, 39:69]
    B_X_test = data[1:, 69:78]

    B_y_train = data[0:1, 39:69]
    B_y_test = data[0:1, 69:78]

    C_X_train = data[1:, 78:108]
    C_X_test = data[1:, 108:117]

    C_y_train = data[0:1, 78:108]
    C_y_test = data[0:1, 108:117]

    D_X_train = data[1:, 117:147]
    D_X_test = data[1:, 147:156]

    D_y_train = data[0:1, 117:147]
    D_y_test = data[0:1, 147:156]

    E_X_train = data[1:, 156:186]
    E_X_test = data[1:, 186:195]

    E_y_train = data[0:1, 156:186]
    E_y_test = data[0:1, 186:195]

    Xtrain = np.transpose(np.hstack((A_X_train, B_X_train, C_X_train, D_X_train, E_X_train)))
    ytrain = np.transpose(np.hstack((A_y_train, B_y_train, C_y_train, D_y_train, E_y_train)))
    Xtest = np.transpose(np.hstack((A_X_test, B_X_test, C_X_test, D_X_test, E_X_test)))
    ytest = np.transpose(np.hstack((A_y_test, B_y_test, C_y_test, D_y_test, E_y_test)))
    trainingSet = np.transpose(np.hstack((trainingSetA, trainingSetB, trainingSetC, trainingSetD, trainingSetE)))
    testSet = np.transpose(np.hstack((testSetA, testSetB, testSetC, testSetD, testSetE)))
    for x in range(len(trainingSet)):
        temp = trainingSet[x][0]
        for y in range(len(trainingSet[x]) - 1):
            trainingSet[x][y] = trainingSet[x][y + 1]
        trainingSet[x][len(trainingSet[x]) - 1] = temp

    for x in range(len(testSet)):
        temp = testSet[x][0]
        for y in range(len(testSet[x]) - 1):
            testSet[x][y] = testSet[x][y + 1]
        testSet[x][len(testSet[x]) - 1] = temp

    y = ytrain.ravel()
    ytrain = np.array(y).astype(int)

    if (classifier == "KNN"):
        predictions = []
        k = 5
        for x in range(len(testSet)):
            neighbours = getNeighbours(trainingSet, testSet[x], k)
            result = getResponse(neighbours)
            predictions.append(result)
        accuracy = getAccuracy(testSet, predictions)
    elif (classifier == "Centroid"):
        centroid_acc = predict(trainX, trainY, testX, testY, 5)
        accuracy = centroid_acc
    elif (classifier == "SVM"):
        svmclassifier = svm.LinearSVC()
        svmclassifier.fit(Xtrain, ytrain)
        predictions = svmclassifier.predict(Xtest)
        actual = ytest
        accuracy = metrics.accuracy_score(actual, predictions) * 100
    elif (classifier == "LogisticRegression"):
        regr = linear_model.LogisticRegression()
        regr.fit(Xtrain, ytrain)
        predictions = regr.predict(Xtest)
        actual = ytest
        accuracy = metrics.accuracy_score(actual, predictions) * 100
    print("accuracy with ", classifier, " is", round(accuracy, 2))
    print("----------------------------------------------")
    return accuracy

def main():



    Y_knn = classification("KNN")
    Y_cent = classification("Centroid")
    Y_svm = classification("SVM")
    Y_regress = classification("LogisticRegression")
if __name__ == '__main__':
    main()