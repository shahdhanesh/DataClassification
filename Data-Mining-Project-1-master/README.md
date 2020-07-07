# Data-Mining-Project-1

## Task A:
In this task we are using HandWrittenLetters.txt data, we are performing Classification for 4 classifiers – KNN, Centroid, SVM and Logistic Regression.<br>
<b>Observation:</b><br>
Average Accuracies (%)<br>
KNN - 91.11<br>
Centroid - 86.67<br>
SVM - 91.11<br>
Logistic Regression - 93.33<br>

Accuracy for Logistic Regression classifier is always highest than SVM, KNN and Centroid. As you can see the accuracy for Logistic Regression classifier is highest and it always falls under range 92-100%.<br>

## Task B:

In this task we are using HandWrittenLetters.txt data, we are performing 5-fold Cross Validation for 4 classifiers – KNN, Centroid, SVM and Logistic Regression.
<br><b>Observation:</b><br>
Average Accuracies (%)<br>
<b>1-Fold</b><br>
KNN - 78.75<br>
Centroid - 87.5<br>
SVM - 100.0<br>
Logistic Regression - 97.50<br>
<b>2-Fold</b><br>
KNN - 75.00<br>
Centroid - 90.0<br>
SVM - 98.95<br>
Logistic Regression - 95.00<br>
<b>3-Fold</b><br>
KNN - 78.05<br>
Centroid - 98.75<br>
SVM - 98.50<br>
Logistic Regression - 96.00<br>
<b>4-Fold</b><br>
KNN - 75.00<br>
Centroid - 93.75<br>
SVM - 93.75<br>
Logistic Regression - 98.50<br>
<b>5-Fold</b><br>
KNN - 68.75<br>
Centroid - 93.75<br>
SVM - 90.75<br>
Logistic Regression - 95.75<br>

1. Accuracy for KNN classifier is lowest in this case than SVM, Logistic Regression and Centroid.
As you can see from the figure, average accuracies for KNN classifier is lowest and it always falls under range 67-80% (Refer Figure 1.2)

2. Accuracy with 2-Fold cross validation is always low.

3. KNN has lowest accuracy.

## Task C: 

In this task we fix 10 classes. We use the data handler to generate training and test data files. We have split the data in 7 parts and named each one as split-1, split-2 and so on. The classes that we fix are based on the input string.
<br><br>After running multiple iterations, we see a trend that the accuracy is highest when the number of training and testing dataset is almost equal. For example, in the output the split-5 has training dataset = 25 and testing dataset = 24.

## Task D: 

This task is similar to Task C. We use the data handler to generate training and test data files. We have split the data in 7 parts and named each one as split-1, split-2 and so on. The input string that we use to fix classes is different from Task C.
<br><br>After running multiple iterations, we conclude that the accuracy is stable and highest when the number of training and testing dataset is almost equal. For example, in the output Split-4 and Split-5 readings are stable and highest as the Training and Testing dataset is almost equal.

## Task E:

In this task, we have made a Data_Handler. In this task we have defined 3 subroutines working on HandWrittenLetters.txt

#### 1. pickData() :

<b>Parameters:</b> filename, class<br>
<b>Description:</b> In this subroutine, we are generating train and test data. The train and test data are generated for the specified array of class numbers with respect to the number of training instances and test instances on the given file.<br>

#### 2. storeData():

<b>Parameters:</b> train_X, train_Y, test_X, test_Y<br>
<b>Description:</b> In this subroutine, we collected the train and test data in to different files namely, trainData.txt and testData.txt respectively. We can use these file and feed into classifier.<br>

#### 3. letter_2_digit_convert()

<b>Parameters:</b> A string of characters<br>
<b>Description:</b> In this subroutine, we convert the characters from the string into their respective class values and return as an array.<br>
