import data_handler
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import Task1

all_accs = []

sample_string = "adgjwryozm"
splits = ["1:34", "1:29", "1:24", "1:19", "1:14", "1:9", "1:4"]
for split in splits:
    print ("Current split %s"%split)
    trainX, trainY, testX, testY = data_handler.splitData2TestTrain(data_handler.pickDataClass('Handwrittenletters.txt',data_handler.letter_2_digit_convert(sample_string)),39,split)


    all_accs.append(Task1.predict(trainX, trainY, testX, testY, 10))

print(all_accs)

x = [1,2,3,4,5,6,7]
# Plot the data
for i in range(len(x)):

   plt.scatter(x[i], all_accs[i],color='black',marker='^')
   plt.plot(x[i], all_accs[i], label= i)
plt.legend(loc='lower right', frameon=False)
# Show the plot
plt.show()