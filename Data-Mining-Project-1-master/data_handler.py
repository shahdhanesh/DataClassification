import math
import sys
import operator
import random


def get_data(filename):
    return pre_processor(file=open(filename))


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
    return data, indexes


def train_test_split(data):
    split_randInt = random.randint(int(len(data) / 2), (len(data) - 5))
    random.shuffle(data)
    train_data = data[0:split_randInt]
    test_data = data[split_randInt:-1]
    return train_data, test_data


def pickDataClass(filename, class_ids):
    '''
	    subroutine-1: pickDataClass(filename, class_ids)
	      filename: char_string specifying the data file to read. For example, 'ATNT_face_image.txt'
	      class_ids:  array that contains the classes to be pick. For example: (3, 5, 8, 9)
	      Returns: an multi-dimension array containing the data (both attribute vectors and class labels)
	               of the selected classes
	      e.g. pickDataClass('ATNTFaceImages400.txt', (3, 5, 8, 9))
    '''
    dataset, class_indexes = get_data(filename)
    picked_data = []
    for i in list(class_ids):
        picked_data.extend([j for j in dataset if j[-1] == str(i)])
    return picked_data


def splitData2TestTrain(filename, number_per_class, test_instances):
    '''
	    subroutine-2: splitData2TestTrain(filename, number_per_class,  test_instances)
	      filename: char_string specifying the data file to read. This can also be an array containing input data.
	      number_per_class: number of data instances in each class (we assume every class has the same number of data instances)
	      test_instances: the data instances in each class to be used as test data.
	                      We assume that the remaining data instances in each class (after the test data instances are taken out)
	                      will be training_instances
	      Return/output: Training_attributeVector(trainX), Training_labels(trainY), Test_attributeVectors(testX), Test_labels(testY)
	      The data should easily fed into a classifier.
	      Example: splitData2TestTrain('Handwrittenletters.txt', 39, 1:20)
	               Use entire 26-class handwrittenletters data. Each class has 39 instances.
	               In every class, first 20 images for testing, remaining 19 images for training
    '''
    if isinstance(filename, str):
        dataset, class_indexes = get_data(filename)
    else:
        dataset = filename
        class_indexes = [item[-1] for item in dataset]

    unique_class_indexes = list(set(class_indexes))
    per_class_instances = []
    for i in unique_class_indexes:
        per_class_instances.append([j for j in dataset if str(j[-1]) == str(i)][:number_per_class])
    test_inst_count = int(test_instances.rsplit(":")[-1])
    trainX = []
    trainY = []
    testX = []
    testY = []
    for item in per_class_instances:
        testX.extend(item[:test_inst_count])
        trainX.extend(item[test_inst_count:])
    trainY = [item[-1] for item in trainX]
    testY = [item[-1] for item in testX]
    return trainX, trainY, testX, testY


# splitData2TestTrain(pickDataClass('Handwrittenletters.txt', (21,22,23,24)), 39, "1:20")
# splitData2TestTrain(pickDataClass('Handwrittenletters.txt', (21,22,23,24)), 9,  "1:20")
# pickDataClass('Handwrittenletters.txt', (3, 5, 8, 9))

def store_data(trainX, trainY, testX, testY):
    '''
        subroutine-3:
       This routine will store (trainX,trainY) into a training data file,
       and store (testX,testY) into a test data file. The format of these files is determined by
       student's choice: could be a Matlab file, a text file, or a file convenient for Python.
       These file should allow the data to be easily read and feed into a classifier.
       During a COMPUTER QUIZ, you use this routine to save the files and submit them as part of the quiz results.
    '''
    train_dict = {"trainX": trainX, "trainY": trainY}
    test_dict = {"testX": testX, "testY": testY}

    file = open("trainData.txt", "w")
    file.write(str(train_dict))
    file.close()

    file = open("testFile.txt", "w")
    file.write(str(test_dict))
    file.close()


# # store_data(splitData2TestTrain(pickDataClass('Handwrittenletters.txt', (21,22,23,24)), 39, "1:20"))
# trainX, trainY, testX, testY = splitData2TestTrain(pickDataClass('Handwrittenletters.txt', (21,22,23,24)), 39, "1:20")
# store_data(trainX, trainY, testX, testY)
# print ("S")

'''
Subroutine-4: "letter_2_digit_convert" that converts a character string to an integer array.
   For example ,letter_2_digit_convert('ACFG') returns array (1, 3, 6, 7).
   A COMPUTER QUIZ problem could be: Pick 5 classes with letters 'great' from the hand-written-letter data, and
     generate training and testing data using first 20 images of each class for training and the rest 19 images for test.
     You will need to use  letter_2_digit_convert('GREAT') to convert to numbers and then subroutine-1 to pick the subset
     of the needed data.
'''


def letter_2_digit_convert(sample):
    return ([ord(item.upper()) - 64 for item in sample])

# letter_2_digit_convert("ASDFRE")
