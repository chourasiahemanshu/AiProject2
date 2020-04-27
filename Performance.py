# pseudo code for performance metrics

#calculate TP, FP, TN, FN

# using these values, calculate Accuracy, Precision, Recall, F1-measure and consfusion matrix
#######################################
#total positive = total ham files
#total negative = total spam files

#toal files = total negative + total positive

#read test results 
# calculate TP, FP, TN, FN
# also create function that cankeep updating values as each file is processed for testing


# accuracy = TP+TN / TP+FP+FN+TN
# precision = TP/ TP+FP
# recall = TP / TP+FN
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)



# using matplotlib
# https://matplotlib.org/3.1.1/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py

# confusion_matrix_data = [[TP, FN],
#                      [FP, TN]]

#columns = ['Ham', 'Spam']
#rows = ['Ham','Spam']

#confusion_matrix = plt.table(cellText = confusion_matrix_data , rowLabels = rows, colLabels = columns)
# plt.title('Confusion Matrix')
# plt.show() 